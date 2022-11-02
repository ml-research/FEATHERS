import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from opacus.grad_sample import register_grad_sampler
from typing import Dict
from utils import get_params


class ParallelOp(nn.Module):

  def __init__(self, C, stride) -> None:
    super().__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.GroupNorm(num_groups=1, num_channels=C, affine=False))
      self._ops.append(op)

  def forward(self, x):
      operation_outs = []
      for op in self._ops:
          out = op(x)
          operation_outs.append(out)
      return torch.stack(operation_outs)

class MixedOp(nn.Module):

    def __init__(self):
        super(MixedOp, self).__init__()
        self.alphas = nn.Parameter(torch.zeros(len(PRIMITIVES)), requires_grad=True)

    def forward(self, x):
        weights = torch.softmax(self.alphas, 0)
        return sum(w * op_out for w, op_out in zip(weights, x))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, mixed_ops_normal, mixed_ops_reduce):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    mixed_op_idx = 0
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        if reduction:
          op = nn.Sequential(ParallelOp(C, stride), mixed_ops_reduce[mixed_op_idx])
        else:
          op = nn.Sequential(ParallelOp(C, stride), mixed_ops_normal[mixed_op_idx])
        mixed_op_idx += 1
        self._ops.append(op)

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, device, in_channels=3, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.device = device

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.GroupNorm(num_groups=1, num_channels=C_curr),
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    self._init_mixed_ops()
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.mixed_ops_normal, self.mixed_ops_reduce)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.device).to(self.device)
    for x, y in zip(get_params(model_new, 'arch'), get_params(self, 'arch')):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _init_mixed_ops(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    self.mixed_ops_normal = [MixedOp() for _ in range(k)]
    self.mixed_ops_reduce = [MixedOp() for _ in range(k)]

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    alphas_normal = torch.stack([mop.alphas.data for mop in self.mixed_ops_normal])
    alphas_reduce = torch.stack([mop.alphas.data for mop in self.mixed_ops_reduce])
    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

@register_grad_sampler(ParallelOp)
def grad_sampler_parallel_op(layer: MixedOp, activations: torch.Tensor, backprops: torch.Tensor):
    return {}

@register_grad_sampler(MixedOp)
def grad_sampler_mixed_op(layer: MixedOp, activations: torch.Tensor, backprops: torch.Tensor):
    grad = torch.einsum('nbcwh,bcwh->nb', activations, backprops)
    ret = {
        layer.alphas: grad
    }
    return ret