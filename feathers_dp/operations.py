import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

TABOPS = {
  'relu_ln_1': lambda in_dim, out_dim: ReluLN(in_dim, out_dim),
  'sigmoid_ln_1': lambda in_dim, out_dim: SigmoidLN(in_dim, out_dim),
  'tanh_ln_1': lambda in_dim, out_dim: TanhLN(in_dim, out_dim),
  'relu_ln_2_reddim': lambda in_dim, out_dim: ReluLNRedDim(in_dim, out_dim),
  'sigmoid_2_reddim': lambda in_dim, out_dim: SimgoidLNRedDim(in_dim, out_dim),
  'tanh_ln_2_reddim': lambda in_dim, out_dim: TanhLNRedDim(in_dim, out_dim),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class TabZero(nn.Module):
  def __init__(self):
    super(TabZero, self).__init__()

  def forward(self, x):
    return x * 0

class ReluLN(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    return torch.relu(self.linear(x))

class SigmoidLN(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    return torch.sigmoid(self.linear(x))

class TanhLN(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    return torch.tanh(self.linear(x))

class ReluLNExpDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int(1.25*in_dim))
    self.linear2 = nn.Linear(int(1.25*in_dim), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    return torch.relu(self.linear2(x))

class SigmoidLNExpDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int(1.25*in_dim))
    self.linear2 = nn.Linear(int(1.25*in_dim), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.sigmoid(x)
    return torch.sigmoid(self.linear2(x))
  
class TanhLNExpDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int(1.25*in_dim))
    self.linear2 = nn.Linear(int(1.25*in_dim), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.tanh(x)
    return torch.tanh(self.linear2(x))

class ReluLNRedDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int((in_dim + out_dim) / 2))
    self.linear2 = nn.Linear(int((in_dim + out_dim) / 2), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    return torch.relu(self.linear2(x))

class SimgoidLNRedDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int((in_dim + out_dim) / 2))
    self.linear2 = nn.Linear(int((in_dim + out_dim) / 2), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.sigmoid(x)
    return torch.tanh(self.linear2(x))

class TanhLNRedDim(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.linear1 = nn.Linear(in_dim, int((in_dim + out_dim) / 2))
    self.linear2 = nn.Linear(int((in_dim + out_dim) / 2), out_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.tanh(x)
    return torch.tanh(self.linear2(x))