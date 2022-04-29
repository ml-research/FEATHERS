import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import PoolBN, StdConvRelu, StdConvTanh, Identity, FactorizedReduce

OPERATIONS = [StdConvTanh, StdConvRelu, Identity]

class MixedOp(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True) -> None:
        super().__init__()
        self.ops = torch.nn.ModuleList()
        self.alpahs = torch.nn.Parameter(1e-3*torch.randn(len(OPERATIONS) + 2), requires_grad=True)
        self.ops.append(PoolBN('avg', C_in, kernel_size, stride, padding, affine))
        self.ops.append(PoolBN('max', C_in, kernel_size, stride, padding, affine))
        for op in OPERATIONS:
            operation = op(C_in, C_out, kernel_size, stride, padding, affine)
            self.ops.append(operation)
    
    def forward(self, x):
        softmaxed_weights = F.softmax(self.alpahs, dim=0)
        return sum(w * op(x) for w, op in zip(softmaxed_weights, self.ops))

    def arch_parameters(self):
        return self.alpahs


class Node(nn.Module):

    def __init__(self, num_prev_nodes, channels, reduction):
        super().__init__()
        self.ops = nn.ModuleList()
        for i in range(num_prev_nodes):
            stride = 1
            if reduction:
                stride = 2 if i < 2 else 1 # reduce size if this is an edge from the first two nodes to this one
            op = MixedOp(channels, channels, 1, stride, 0)
            self.ops.append(op)

    def forward(self, x):
        out = sum([op(x_prev) for x_prev, op in zip(x, self.ops)])
        return out

    def arch_parameters(self):
        params = [op.arch_parameters() for op in self.ops]
        return params


class Cell(torch.nn.Module):
    def __init__(self, channels_p, channels_pp, channels, reduction, reduction_p, node_nr, sum_output=True) -> None:
        super().__init__()
        if reduction_p:
            self.preporcess_prev_prev = FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preporcess_prev_prev = StdConvRelu(channels_pp, channels, 1, 1, 0, affine=False)
        self.preprocess_prev = StdConvRelu(channels_p, channels, 1, 1, 0, affine=False)
        self.nodes = torch.nn.ModuleList()
        self.sum_output = sum_output
        for i in range(2, node_nr + 2):
            node = Node(i, channels, reduction)
            self.nodes.append(node)
        

    def arch_parameters(self):
        params = []
        for node in self.nodes:
            params += node.arch_parameters()
        return params

    def forward(self, x_prev, x_prev_prev):
        last_out = [self.preporcess_prev_prev(x_prev_prev), self.preprocess_prev(x_prev)]
        out = None
        # for each node, compute representation
        for node in self.nodes:
            out = node(last_out)
            last_out.append(out)
        if self.sum_output:
            return sum(last_out)
        else:
            return torch.cat(last_out[2:], dim=1)


class Classifier(torch.nn.Module):

    def __init__(self, num_classes, criterion, layers, in_channels, channels, cell_node_nr=5, sum_cell_output=False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cell_node_nr = cell_node_nr
        self.criterion = criterion
        self.layers = layers
        self.in_channels = in_channels
        self.channels = channels
        self.sum_cell_output = sum_cell_output
        self.c_cur = self.channels * 3 # build 3x of input-channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.c_cur)
        )
        self.cells = torch.nn.ModuleList()
        self._init_cells()
        
    def _init_cells(self):
        channels_pp, channels_p, self.c_cur = self.c_cur, self.c_cur, self.channels
        reduction_p, reduction = False, False
        for i in range(1, self.layers):
            reduction_p, reduction = reduction, False
            # build suerp-net using alternating cell-types (reduction cells and normal cells)
            # if i % 2 == 1:
            if i in [self.layers//3, 2*self.layers//3]:
                self.c_cur *= 2
                reduction = True
            cell = Cell(channels_p, channels_pp, self.c_cur, reduction, reduction_p, self.cell_node_nr, self.sum_cell_output)
            self.cells.append(cell)
            # if we concatenate the nodes outputs in a cell, we have to adjust the channel-number
            if not self.sum_cell_output:
                c_cur_out = self.c_cur * self.cell_node_nr
            else:
                c_cur_out = self.c_cur
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, self.num_classes)
        #self.bn = nn.BatchNorm1d(self.num_classes)

    def forward(self, x):
        x_prev_prev = x_prev = self.stem(x)
        for i in range(len(self.cells)):
            cell = self.cells[i]
            x_prev_prev, x_prev = x_prev, cell(x_prev, x_prev_prev)
        x = self.gap(x_prev)
        x = x.view(x.shape[0], -1) # flatten
        x = self.linear(x)
        return x
        # return F.softmax(x, dim=1)

    def net_loss(self, x, y):
        logits = self(x)
        return self.criterion(logits, y)

    def arch_parameters(self):
        params = []
        for cell in self.cells:
            params += cell.arch_parameters()
        return params