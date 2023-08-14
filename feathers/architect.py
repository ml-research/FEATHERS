import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from numpy.linalg import eigvals

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, model_momentum, model_weight_decay, arch_lr, 
                arch_weight_decay, device):
    self.device = device
    self.network_momentum = model_momentum
    self.network_weight_decay = model_weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=arch_lr, betas=(0.5, 0.999), weight_decay=arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.to(self.device)

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def update_hyperparameters(self, hyperparams):
    for g in self.optimizer.param_groups:
      g['lr'] = hyperparams['arch_learning_rate']
      g['weight_decay'] = hyperparams['arch_weight_decay']

    self.network_momentum = hyperparams['momentum']
    self.network_weight_decay = hyperparams['weight_decay']

  def compute_Hw(self, input_valid, target_valid):
      self.zero_grads(self.model.parameters())
      self.zero_grads(self.model.arch_parameters())

      loss = self.model._loss(input_valid, target_valid)
      self.hessian = self._hessian(loss, self.model.arch_parameters())
      return self.hessian
  
  def compute_eigenvalues(self):
      #hessian = self.compute_Hw(input, target)
      if self.hessian is None:
          raise ValueError
      return eigvals(self.hessian.cpu().data.numpy())
  
  def zero_grads(self, parameters):
      for p in parameters:
          if p.grad is not None:
              p.grad.detach_()
              p.grad.zero_()
              #if p.grad.volatile:
              #    p.grad.data.zero_()
              #else:
              #    data = p.grad.data
              #    p.grad = Variable(data.new().resize_as_(data).zero_())

  def _hessian(self, outputs, inputs, out=None, allow_unused=False,
               create_graph=False):
      #assert outputs.data.ndimension() == 1
      if torch.is_tensor(inputs):
          inputs = [inputs]
      else:
          inputs = list(inputs)
      n = sum(p.numel() for p in inputs)
      if out is None:
          out = Variable(torch.zeros(n, n)).type_as(outputs)
      ai = 0
      for i, inp in enumerate(inputs):
          [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                       allow_unused=allow_unused)
          grad = grad.contiguous().view(-1) + self.weight_decay*inp.view(-1)
          #grad = outputs[i].contiguous().view(-1)
          for j in range(inp.numel()):
              # print('(i, j): ', i, j)
              if grad[j].requires_grad:
                  row = self.gradient(grad[j], inputs[i:], retain_graph=True)[j:]
              else:
                  n = sum(x.numel() for x in inputs[i:]) - j
                  row = Variable(torch.zeros(n)).type_as(grad[j])
                  #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)
              out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
              if ai + 1 < n:
                  out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
              del row
              ai += 1
          del grad
      return out