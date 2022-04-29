import torch
import torch.nn as nn

class LeakyReLUFCLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.lrelu(x)

class TanhFCLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        return self.tanh(x)

class FCBNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.odim = out_dim

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConvRelu(nn.Module):
    """
    Standard conv: Conv - Relu - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class StdConvTanh(nn.Module):
    """
    Standard conv: Conv - Tanh - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class Identity(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True) -> None:
        super().__init__()
        if stride == 1:
            self.op = None
        else:
            self.op = FactorizedReduce(C_in, C_out, affine)

    def forward(self, x):
        if self.op is not None:
            return self.op(x)
        else:
            return x