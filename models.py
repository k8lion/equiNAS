from sys import get_coroutine_origin_tracking_depth
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import numpy as np
import copy
import uuid
#import escnn

def rotate_4(x: torch.Tensor, r: int) -> torch.Tensor:
    return x.rot90(r, dims=(-2, -1))

def rotate_n(x: torch.Tensor, r: int, n: int) -> torch.Tensor:
    if r == 0:
        roty = x
    elif r/n == 1/2:
        roty = rotate_4(x, 2)
    elif r/n == 1/4:
        roty = rotate_4(x, 1)
    elif r/n == 3/4:
        roty = rotate_4(x, 3)
    else:
        #roty = rot(x, 2*r*np.pi/n, type(x))
        shape = x.shape
        roty = tvF.rotate(x.reshape(x.size(1), -1, x.size(-2), x.size(-1)), 360*r/n).reshape(shape)
    return roty

def rotatestack_n(y: torch.Tensor, r: int, n: int, order = None) -> torch.Tensor:
    if order is None:
        order = list(range(n))
    assert len(y.shape) >= 3
    assert y.shape[-3] == n
    assert len(order) == n

    roty = rotate_n(y, r, n)
    roty = torch.stack([torch.select(roty, -3, (n-r+i)%n) for i in order], dim=-3) 
    return roty

def transform(y: torch.Tensor, g: tuple) -> torch.Tensor:
  
    assert len(y.shape) >= 2

    f, r = g

    y = y.clone()

    if f == 1:
      y = torch.flip(y, dims=(-2,))
    y = y.rot90(r, dims=(-2, -1))
    return y


"""
group action on images

inputs:
x: A x B x C x D
r: rotational aspect of group member
n_r: rotational aspects of group
f: flip aspect of group member
n_f: flip aspects of group

output: A x B x C x D, C x D data rotated/flipped
"""
def rotateflip_n(x: torch.Tensor, r: int, n_r: int, f: int, n_f: int, flipfirst: bool = False) -> torch.Tensor:
    if n_f == 2 and n_r == 4:
        return transform(x, (f,r))
    if n_f == 2 and n_r == 2:
        return transform(x, (f,r*2))
    if r == 0:
        roty = x
    elif r/n_r == 1/2:
        roty = rotate_4(x, 2)
    elif r/n_r == 1/4:
        roty = rotate_4(x, 1)
    elif r/n_r == 3/4:
        roty = rotate_4(x, 3)
    else:
        shape = x.shape
        roty = tvF.rotate(x.reshape(x.size(1), -1, x.size(-2), x.size(-1)), 360*r/n_r).reshape(shape)
    if not flipfirst and n_f == 2 and f == 1:
        #if n_f == 2 and f == 1:
        roty = torch.flip(roty, (-1,))
    return roty


def transform_n(y: torch.Tensor, g: tuple, n: int) -> torch.Tensor:
    assert len(y.shape) >= 2

    f, r = g
    y = y.clone()
    if f == 1:
      y = torch.flip(y, dims=(-2,))
    if (r/n*4).is_integer():
        y = y.rot90(int(r/n*4), dims=(-2, -1))
    else:
        shape = y.shape
        y = tvF.rotate(y.reshape(y.size(1), -1, y.size(-2), y.size(-1)), 360*r/n).reshape(shape)
    return y

def transform_pnm(y: torch.Tensor, g: tuple, n: int) -> torch.Tensor:
    assert len(y.shape) >= 3
    assert y.shape[-3] == 2*n

    f, r = g
    y = transform_n(y, g, n)

    y = y.reshape(*y.shape[:-3], 2, n, *y.shape[-2:])
    if f:
      y = torch.index_select(y, -3, torch.LongTensor([0]+list(range(n-1,0,-1))).to(y.device))
      y = torch.flip(y, dims=(-4,))
    y = torch.roll(y, r, dims=-3)
    y = y.reshape(*y.shape[:-4], 2*n, *y.shape[-2:])
    return y

"""
group action on lifted activations/weights

inputs:
y: A x B x |G| x C x D
r: rotational aspect of group member
n_r: rotational aspects of group
f: flip aspect of group member
n_f: flip aspects of group

output: A x B x |G| x C x D, 3rd dimension rolled and C x D data transformed for each member of G
"""
def rotateflipstack_n(y: torch.Tensor, r: int, n_r: int, f: int, n_f: int, order = None) -> torch.Tensor:
    if n_f == 1:
        return rotatestack_n(y, r, n_r)
    return transform_pnm(y, (f,r), n_r)

def adapt(parentweight: torch.Tensor, parentg, childg, inchannels, outchannels):
    parentorder = [int('{:0{width}b}'.format(n, width=sum(parentg))[::-1], 2) for n in range(groupsize(parentg))]
    order = [parentorder[groupdifference(parentg, childg)*i:(i+1)*groupdifference(parentg, childg)] for i in [int('{:0{width}b}'.format(n, width=sum(childg))[::-1], 2) for n in range(groupsize(childg))]]
    order=[[0,1],[2,3]]
    if childg == 0:
        order = order[0]
    splito = int(outchannels/groupdifference(parentg, childg))
    outchannelorder = sum([list(range(i,outchannels,splito)) for i in range(splito)], [])
    spliti = int(inchannels/groupdifference(parentg, childg))
    inchannelorder = sum([list(range(i,inchannels,spliti)) for i in range(spliti)], [])
    weight = torch.cat([rotate_n(torch.cat([parentweight[:,:,order[i]] for i in range(len(order))], dim=0), j, groupsize(parentg)) for j in range(groupdifference(parentg, childg))], dim=1)[outchannelorder]
    if childg == 0:
        weight = torch.unsqueeze(weight, dim = 2)
    return weight[:, inchannelorder]

def groupsize(g: tuple):
    return 2**max(sum(g),0)

def subgroupsize(g: tuple, i: int):
    return 2**max(g[i],0)

def groupdifference(g1: tuple, g2: tuple):
    for i in range(len(g1)):
        assert g1[i] >= g2[i]
    return 2**(sum(g1)-sum(g2))

class LiftingConv2d(torch.nn.Module):

    def __init__(self, group: tuple, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):

        super(LiftingConv2d, self).__init__()

        self.group = group
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        #self.order = [int('{:0{width}b}'.format(n, width=self.group[1])[::-1], 2) for n in range(groupsize(self.group))]
        self.order = range(groupsize(self.group))

        self.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1 / (out_channels * in_channels)**(
                1/2), size=(out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def replicate(self, group:tuple, ordered: bool = False):
        filter, bias = self.build_filter()
        out_channels = int(self.out_channels*groupsize(self.group)/groupsize(group))
        
        if ordered:
            order1 = [int('{:0{width}b}'.format(n, width=int(np.log2(groupsize(self.group))))[::-1], 2) for n in range(groupsize(self.group))]
            order2 = [int('{:0{width}b}'.format(n, width=int(np.log2(groupsize(group))))[::-1], 2) for n in range(groupsize(group))]
        else:
            order1 = range(groupsize(self.group))
            order2 = range(groupsize(group))

        filter = filter.clone()[:,order1].reshape(self.out_channels * groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)
        filter = filter.reshape(out_channels, groupsize(group), self.in_channels, self.kernel_size, self.kernel_size)[:,order2]

        child = LiftingConv2d(group, self.in_channels, out_channels, self.kernel_size, self.padding, self.bias is not None)

        child.weight.data = filter[:, 0]
        if bias is not None:
            bias = bias.clone()[:,order1].reshape(self.out_channels * groupsize(self.group)).reshape(out_channels, groupsize(group))[:,order2]
            child.bias.data = bias[:, 0]

        return child
  
    def test_filter_x(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, groupsize(self.group))
        assert _filter.shape == (
                self.out_channels, groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(
                self.out_channels * groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)

        return _filter, x

    def build_filter(self) -> torch.Tensor:
        if self.group[0] == 1:
            n_r = subgroupsize(self.group, 1)
            _filter = torch.stack([rotateflip_n(self.weight, i%n_r, n_r, i//n_r, 2)
                            for i in self.order], dim=-4)
        else:
            _filter = torch.stack([rotate_n(self.weight, i, groupsize(self.group))
                            for i in self.order], dim=-4)

        if self.bias is not None:
            _bias = torch.stack([self.bias for _ in range(groupsize(self.group))], dim=1)
        else:
            _bias = None

        return _filter, _bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, groupsize(self.group))
        assert _filter.shape == (
                self.out_channels, groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(
                self.out_channels * groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * groupsize(self.group))

        out = torch.conv2d(x, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=_bias)

        return out.view(-1, self.out_channels, groupsize(self.group), out.shape[-2], out.shape[-1])

class GroupConv2d(torch.nn.Module):

    def __init__(self, group: tuple, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True, perm=list(range(8))):
        
        super(GroupConv2d, self).__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        self.weight = torch.nn.Parameter(torch.normal(mean = 0.0, std = 1 / (out_channels * in_channels)**(1/2), 
                size=(out_channels, in_channels, groupsize(self.group), kernel_size, kernel_size)), requires_grad=True)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
    
    def replicate(self, group:tuple, ordered: bool = False):
        filter, bias = self.build_filter()
        in_channels = int(self.in_channels*groupsize(self.group)/groupsize(group))
        out_channels = int(self.out_channels*groupsize(self.group)/groupsize(group))

        if ordered:
            order1 = [int('{:0{width}b}'.format(n, width=int(np.log2(groupsize(self.group))))[::-1], 2) for n in range(groupsize(self.group))]
            order2 = [int('{:0{width}b}'.format(n, width=int(np.log2(groupsize(group))))[::-1], 2) for n in range(groupsize(group))]
        else:
            order1 = range(groupsize(self.group))
            order2 = range(groupsize(group))

        filter = filter.clone()[:, order1][:,:,:,order1].reshape(self.out_channels * groupsize(self.group), self.in_channels * groupsize(self.group), self.kernel_size, self.kernel_size)
        filter = filter.reshape(out_channels, groupsize(group), in_channels, groupsize(group), self.kernel_size, self.kernel_size)[:, order2][:,:,:, order2]

        child = GroupConv2d(group, in_channels, out_channels, self.kernel_size, self.padding, self.bias is not None)
        child.weight.data = filter[:, 0]
        if bias is not None:
            bias = bias.clone()[:,order1].reshape(self.out_channels * groupsize(self.group)).reshape(out_channels, groupsize(group))[:,order2]
            child.bias.data = bias[:, 0]

        return child
  
    def build_filter(self) -> torch.Tensor:
        if self.group[0] == 1:
            order = [(i,j) for j in range(subgroupsize(self.group, 0)) for i in range(subgroupsize(self.group, 1))]
            _filter = torch.stack([rotateflipstack_n(self.weight, i, subgroupsize(self.group, 1), j, subgroupsize(self.group, 0)) for (i,j) in order], dim = -5)
        else:
            _filter = torch.stack([rotatestack_n(self.weight, i, groupsize(self.group)) for i in range(subgroupsize(self.group, 1))], dim = -5)

        if self.bias is not None:
            _bias = torch.stack([self.bias for _ in range(groupsize(self.group))], dim = 1)
        else:
            _bias = None

        return _filter, _bias

    def test_filter_x(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, groupsize(self.group))
        assert _filter.shape == (self.out_channels, groupsize(self.group), self.in_channels, groupsize(self.group), self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(self.out_channels * groupsize(self.group), self.in_channels * groupsize(self.group), self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * groupsize(self.group))

        _x = x.clone().view(x.shape[0], self.in_channels*groupsize(self.group), x.shape[-2], x.shape[-1])

        return _filter, _x, _bias


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, groupsize(self.group))
        assert _filter.shape == (self.out_channels, groupsize(self.group), self.in_channels, groupsize(self.group), self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(self.out_channels * groupsize(self.group), self.in_channels * groupsize(self.group), self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * groupsize(self.group))
        #print("gc", x.shape, self.in_channels, self.group)
        x = x.view(x.shape[0], self.in_channels*groupsize(self.group), x.shape[-2], x.shape[-1])

        out = torch.conv2d(x, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=_bias)

        return out.view(-1, self.out_channels, groupsize(self.group), out.shape[-2], out.shape[-1])

class Reshaper(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, in_groupsize: int, out_groupsize: int, ordered = False):

        super(Reshaper, self).__init__()
        print("Reshaper", in_groupsize,in_channels,out_groupsize,out_channels)
        assert in_groupsize*in_channels == out_groupsize*out_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_groupsize = in_groupsize
        self.out_groupsize = out_groupsize
        if ordered:
            self.in_order = [int('{:0{width}b}'.format(n, width=int(np.log2(in_groupsize)))[::-1], 2) for n in range(in_groupsize)]
            self.out_order = [int('{:0{width}b}'.format(n, width=int(np.log2(out_groupsize)))[::-1], 2) for n in range(out_groupsize)]
        else:
            self.in_order = range(in_groupsize)
            self.out_order = range(out_groupsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == self.in_groupsize
        out = x[:,:,self.in_order].view(-1, self.out_channels, self.out_groupsize, x.shape[-2], x.shape[-1])[:,:,self.out_order]
        assert out.shape[1] == self.out_channels
        assert out.shape[2] == self.out_groupsize
        #if self.out_groupsize == 1:
            #print(out.shape)
        return out

class TDRegEquiCNN(torch.nn.Module):
    
    def __init__(self, gs = [(0,0) for _ in range(6)], parent = None, ordered = False, lr = 0.1):
        
        super(TDRegEquiCNN, self).__init__()

        self.uuid = uuid.uuid4()
        if parent is not None:
            self.parent = parent.uuid
        else:
            self.parent = None
        self.gs = gs
        self.channels = [24, 48, 48, 96, 96, 64]
        self.kernels = [7, 5, 5, 5, 5, 5]
        self.paddings = [1, 2, 2, 2, 2, 1]
        self.blocks = torch.nn.ModuleList([])
        self.ordered = ordered
        self.architect(parent)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)


    def architect(self, parent = None):
        reshaper = None
        init = (parent is None)
        if not init and self.gs == parent.gs:
            self.blocks = copy.deepcopy(parent.blocks)
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            return 
        self.blocks.append(torch.nn.Sequential(
                LiftingConv2d(self.gs[0], 1, int(self.channels[0]/groupsize(self.gs[0])), self.kernels[0], self.paddings[0], bias=True),
                torch.nn.BatchNorm3d(int(self.channels[0]/groupsize(self.gs[0]))),
                torch.nn.ReLU(inplace=True)
            )
        )
        if not init:
            self.blocks[0]._modules["0"] = parent.blocks[0]._modules["0"].replicate(self.gs[0], ordered = self.ordered)
        if self.gs[1] != self.gs[0]:
            reshaper = Reshaper(in_channels=int(self.channels[0]/groupsize(self.gs[0])), out_channels=int(self.channels[0]/groupsize(self.gs[1])), in_groupsize=groupsize(self.gs[0]), out_groupsize=groupsize(self.gs[1]), ordered=self.ordered)
            self.blocks[0].add_module(name="reshaper", module = reshaper)

        
        for i in range(1, len(self.gs)):
            self.blocks.append(torch.nn.Sequential(
                GroupConv2d(self.gs[i], int(self.channels[i-1]/groupsize(self.gs[i])), int(self.channels[i]/groupsize(self.gs[i])), self.kernels[i], self.paddings[i], bias=True),
                torch.nn.BatchNorm3d(int(self.channels[i]/groupsize(self.gs[i]))),
                torch.nn.ReLU(inplace=True)
                )
            )
            if i == 1 or i == 3:
                self.blocks[i].add_module(name="pool", module = torch.nn.AvgPool3d((1,3,3), (1,2,2), padding=(0,1,1)))

            if not init and self.gs[i] == parent.gs[i]:
                self.blocks[i] = copy.deepcopy(parent.blocks[i])
            elif not init:   
                self.blocks[i]._modules["0"] = parent.blocks[i]._modules["0"].replicate(self.gs[i], ordered = self.ordered)

            if i < len(self.gs)-1:
                if self.gs[i+1] != self.gs[i]:
                    reshaper = Reshaper(in_channels=int(self.channels[i]/groupsize(self.gs[i])), out_channels=int(self.channels[i]/groupsize(self.gs[i+1])), in_groupsize=groupsize(self.gs[i]), out_groupsize=groupsize(self.gs[i+1]), ordered=self.ordered)
                    self.blocks[i].add_module(name="reshaper", module = reshaper)

        if init:
            self.blocks.append(torch.nn.AvgPool3d((groupsize(self.gs[-1]),5,5), (1,1,1), padding=(0,0,0)))
            self.full1 = torch.nn.Sequential(
                torch.nn.Linear(int(self.channels[-1]/groupsize(self.gs[-1])), 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
            )
            self.full2 = torch.nn.Linear(64, 10)
        else:
            self.blocks.append(copy.deepcopy(parent.blocks[-1]))
        
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            if self.gs[-1] != parent.gs[-1]:
                self.blocks[-1] = torch.nn.AvgPool3d((groupsize(self.gs[-1]),5,5), (1,1,1), padding=(0,0,0))
                self.full1._modules["0"] = torch.nn.Linear(int(self.channels[-1]/groupsize(self.gs[-1])), 64)
                print(parent.full1._modules["0"])
                self.full1._modules["0"].weight.data = torch.repeat_interleave(parent.full1._modules["0"].weight.data, groupdifference(parent.gs[-1], self.gs[-1]), dim=1)/groupdifference(parent.gs[-1], self.gs[-1])
                if parent.full1._modules["0"].bias is not None:
                    self.full1._modules["0"].bias.data = parent.full1._modules["0"].bias.data.clone()
        
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = self.full1(x.reshape(x.shape[0], -1))
        return self.full2(x)

    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(len(self.gs[0])):
            for i in range(1, self.gs[-1][d]+1):
                g = list(self.gs[-1])
                g[d] -= i
                child = self.offspring(len(self.gs)-1, tuple(g))
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                g = list(self.gs[i])
                child = self.offspring(i-1, self.gs[i])
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        return candidates

    def offspring(self, i, G):
        gs = [g for g in self.gs]
        if i >= 0:
            gs[i] = G
        child = TDRegEquiCNN(gs = gs, parent=self, ordered = self.ordered)
        return child


class SkipBlock(torch.nn.Module):
    def __init__(self, *args):
        super(SkipBlock, self).__init__()
        self.before = torch.nn.Sequential(*args)
        self.skip = None
        self.after = torch.nn.Sequential()
        #self._modules = self.before._modules
    
    def forward(self, input):
        out = input.clone()
        out = self.before(out)
        out_ = input.clone()
        for k in self.before._modules.keys():
            out_ = self.before._modules[k](out_)
            #print("before", k, self.before._modules[k], out_.shape)
        #print("before skip add:", out.shape)
        if self.skip is not None:
            #print("skip", self.skip)
            out += self.skip(input)
        else:
            out += input
        out_ = out.clone()
        for k in self.after._modules.keys():
            out_ = self.after._modules[k](out_)
            #print("after", k, self.after._modules[k], out_.shape)
        #print("after skip add:", self.after(out).shape)
        return out_

class SkipEquiCNN(torch.nn.Module):
    
    def __init__(self, gs = [(0,2) for _ in range(8)], parent = None, ordered = False, lr = 0.1, stages = 2, basechannels = 4, superspace = (1,4)):
        
        super(SkipEquiCNN, self).__init__()

        self.uuid = uuid.uuid4()
        if parent is not None:
            self.parent = parent.uuid
        else:
            self.parent = None
        self.gs = gs
        #self.channels = [64, 64, 64, 128, 128, 128]
        #self.kernels = [7, 5, 5, 5, 5, 5]
        #self.paddings = [1, 2, 2, 2, 2, 2]
        self.channels = [groupsize(superspace)*basechannels*2**i for i in range(stages) for _ in range(len(gs)//stages)]
        print(self.channels)
        self.kernels = [5 for _ in range(len(self.channels))]
        self.paddings = [2 for _ in range(len(self.channels))]
        self.blocks = torch.nn.ModuleList([])
        self.ordered = ordered
        self.architect(parent)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)


    def architect(self, parent = None):
        print("Architect", self.gs)
        reshaper = None
        init = (parent is None)
        if not init and self.gs == parent.gs:
            self.blocks = copy.deepcopy(parent.blocks)
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            return 
        self.blocks.append(torch.nn.Sequential(
                LiftingConv2d(self.gs[0], 1, int(self.channels[0]/groupsize(self.gs[0])), self.kernels[0], self.paddings[0], bias=True),
                torch.nn.BatchNorm3d(int(self.channels[0]/groupsize(self.gs[0]))),
                torch.nn.ReLU(inplace=True)
            )
        )
        if not init:
            self.blocks[0]._modules["0"] = parent.blocks[0]._modules["0"].replicate(self.gs[0], ordered = self.ordered)
        if self.gs[1] != self.gs[0]:
            reshaper = Reshaper(in_channels=int(self.channels[0]/groupsize(self.gs[0])), out_channels=int(self.channels[0]/groupsize(self.gs[1])), in_groupsize=groupsize(self.gs[0]), out_groupsize=groupsize(self.gs[1]), ordered=self.ordered)
            self.blocks[0].add_module(name="reshaper", module = reshaper)

        
        for i in range(1, len(self.gs)):
            newblock = SkipBlock(
                GroupConv2d(self.gs[i], int(self.channels[i-1]/groupsize(self.gs[i])), int(self.channels[i]/groupsize(self.gs[i])), self.kernels[i], self.paddings[i], bias=True),
                torch.nn.BatchNorm3d(int(self.channels[i]/groupsize(self.gs[i])))
                )
            newblock.after = torch.nn.ReLU(inplace=True)
            self.blocks.append(newblock)

            if self.channels[i-1] != self.channels[i]:
                self.blocks[i].skip = GroupConv2d(self.gs[i], int(self.channels[i-1]/groupsize(self.gs[i])), int(self.channels[i]/groupsize(self.gs[i])), 1, 0, bias=True)
            
            if i%(len(self.channels)//4) == 0 and i != len(self.channels)-1:
                self.blocks[i].after.add_module(name="pool", module = torch.nn.AvgPool3d((1,3,3), (1,2,2), padding=(0,1,1)))

            if not init and self.gs[i] == parent.gs[i]:
                self.blocks[i] = copy.deepcopy(parent.blocks[i])
            elif not init:   
                self.blocks[i].before._modules["0"] = parent.blocks[i].before._modules["0"].replicate(self.gs[i], ordered = self.ordered)

            if i < len(self.gs)-1:
                if self.gs[i+1] != self.gs[i]:
                    reshaper = Reshaper(in_channels=int(self.channels[i]/groupsize(self.gs[i])), out_channels=int(self.channels[i]/groupsize(self.gs[i+1])), in_groupsize=groupsize(self.gs[i]), out_groupsize=groupsize(self.gs[i+1]), ordered=self.ordered)
                    self.blocks[i].after.add_module(name="reshaper", module = reshaper)

        if init:
            self.blocks.append(torch.nn.AvgPool3d((groupsize(self.gs[-1]),4,4), (1,1,1), padding=(0,0,0)))
            self.full1 = torch.nn.Sequential(
                torch.nn.Linear(int(self.channels[-1]/groupsize(self.gs[-1])), 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
            )
            self.full2 = torch.nn.Linear(64, 10)
        else:
            self.blocks.append(copy.deepcopy(parent.blocks[-1]))
        
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            if self.gs[-1] != parent.gs[-1]:
                self.blocks[-1] = torch.nn.AvgPool3d((groupsize(self.gs[-1]),4,4), (1,1,1), padding=(0,0,0))
                self.full1._modules["0"] = torch.nn.Linear(int(self.channels[-1]/groupsize(self.gs[-1])), 64)
                #print(parent.full1._modules["0"])
                self.full1._modules["0"].weight.data = torch.repeat_interleave(parent.full1._modules["0"].weight.data, groupdifference(parent.gs[-1], self.gs[-1]), dim=1)/groupdifference(parent.gs[-1], self.gs[-1])
                if parent.full1._modules["0"].bias is not None:
                    self.full1._modules["0"].bias.data = parent.full1._modules["0"].bias.data.clone()
        
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = self.full1(x.reshape(x.shape[0], -1))
        return self.full2(x)

    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(len(self.gs[0])):
            for i in range(1, self.gs[-1][d]+1):
                g = list(self.gs[-1])
                g[d] -= i
                child = self.offspring(len(self.gs)-1, tuple(g))
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                child = self.offspring(i-1, self.gs[i])
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        return candidates

    def offspring(self, i, G):
        gs = [g for g in self.gs]
        if i >= 0:
            gs[i] = G
        child = SkipEquiCNN(gs = gs, parent=self, ordered = self.ordered)
        return child

def order(subgroupsize: int):
    return [int('{:0{width}b}'.format(n, width=int(np.log2(subgroupsize)))[::-1], 2) for n in range(subgroupsize)]

class MixedLiftingConv2d(torch.nn.Module):
    def __init__(self, group: tuple, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True, baseline: bool = False, prior: bool = True, discrete: bool = False, norm: bool = True, skip: bool = False):
        super(MixedLiftingConv2d, self).__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels * groupsize(group)
        self.in_channels = in_channels
        self.alphas = torch.nn.Parameter(torch.zeros(np.prod([g+1 for g in group])+1), requires_grad=True)
        if discrete:
            self.alphas.data[:-2] = -np.inf
        elif prior: 
            self.alphas.data[:-2] = -2
        if baseline:
            self.alphas.data[:] = -np.inf
            if prior:
                self.alphas.data[-2] = 0
            else:
                self.alphas.data[0] = 0
            self.alphas.data[-1] = 0
        if skip:
            self.alphas.data[-1] = 0
        else:
            self.alphas.data[-1] = -np.inf
        self.norm = norm
        self.norms = torch.nn.Parameter(torch.zeros(np.prod([g+1 for g in group])+1), requires_grad=False)
        self.weights = torch.nn.ParameterList()
        if bias:
            self.bias = torch.nn.ParameterList()
        else:
            self.bias = None
        self.groups = []
        self.inchannelorders = []
        self.inchannelapply = []
        self.outchannelorders = []
        for i in range(group[0]+1):
            for j in range(group[1]+1):
                g = (i,j)
                in_c = self.in_channels
                out_c = int(self.out_channels/groupsize(g))
                weights = torch.nn.Parameter(torch.normal(mean = 0.0, std = 1 / (out_c * in_c)**(1/2), 
                    size=(out_c, in_c, kernel_size, kernel_size)), requires_grad=True)
                self.norms.data[len(self.weights)] = torch.linalg.norm(weights)
                self.weights.append(weights)
                self.groups.append(g)
                if not discrete and self.group == (0,2) and g != (0,2):
                    self.outchannelorders.append(sum([[4*c,4*c+2,4*c+1,4*c+3] for c in range(int(self.out_channels/4))], start=[]))
                else:
                    self.outchannelorders.append(list(range(self.out_channels)))
                if bias:
                    self.bias.append(torch.nn.Parameter(torch.zeros(out_c), requires_grad=True))
        skip_weights = torch.zeros(size=(0,))
        if self.in_channels != self.out_channels:
            self.alphas.data[-1] = -np.inf
        skip_weights = torch.nn.Parameter(skip_weights, requires_grad=True)
        self.weights.append(skip_weights)
        if bias:
            self.bias.append(torch.nn.Parameter(torch.zeros(self.out_channels), requires_grad=True))
        self.groups.append((-1,-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alphas = torch.softmax(self.alphas, dim=0)

        if self.out_channels % self.in_channels == 0 and alphas[-1] > 0:
            out = alphas[-1]*torch.tile(x, dims=(1,self.out_channels//self.in_channels,1,1))
        elif len(x.shape)>1:
            out = torch.zeros(x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]).to(x.device)

        for layer in range(len(self.groups)-1):
            if alphas[layer] > 0:
                if len(x.shape)>1 and self.norm:
                    weights = self.weights[layer]/torch.linalg.norm(self.weights[layer])*self.norms[layer]
                else:
                    weights = self.weights[layer]

                if self.groups[layer][0] == 1:
                    order = [(i,j) for j in range(subgroupsize(self.groups[layer], 0)) for i in range(subgroupsize(self.groups[layer], 1))]
                    _filter = torch.stack([rotateflip_n(weights, i, subgroupsize(self.groups[layer], 1), j, subgroupsize(self.groups[layer], 0)) for (i,j) in order], dim = -5)
                else:
                    _filter = torch.stack([rotate_n(weights, i, groupsize(self.groups[layer])) for i in range(subgroupsize(self.groups[layer], 1))], dim = -5)

                if self.bias is not None:
                    _bias = torch.stack([self.bias[layer] for _ in range(groupsize(self.groups[layer]))], dim = 1)
                    _bias = _bias.reshape(self.out_channels)
                else:
                    _bias = None

                _filter = _filter.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
                # print(self.outchannelorders[layer])
                _filter = _filter[self.outchannelorders[layer]]

                if len(x.shape)<=1:
                    return _filter

                y = torch.conv2d(x, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=_bias)
                
                out += alphas[layer]*y

        return out
    
    def countparams(self):
        return sum([torch.numel(w) for a, w in zip(torch.softmax(self.alphas, dim=0),self.weights) if a > 0])

    def regularization_loss(self, L2=True, reg_conv=0.0, reg_group=0.0, reg_alphas=0.0):
        weightsum = 0.0
        alphas = torch.softmax(self.alphas, dim=0)
        coeffs = torch.ones_like(alphas)*reg_group
        coeffs[0] = reg_conv
        for i in range(len(alphas)-1):
            if alphas[i] > 0:
                if L2:
                    weightsum += coeffs[i]*self.weights.pow(2).sum()
                else:
                    weightsum += coeffs[i]*self.weights.abs().sum()
        if L2:
            weightsum += reg_alphas*self.alphas.pow(2).sum()
        else:
            weightsum += reg_alphas*self.alphas.abs().sum()
        return weightsum


class MixedGroupConv2d(torch.nn.Module):
    def __init__(self, group: tuple, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True, baseline: bool = False, prior: bool = True, discrete: bool = False, norm: bool = True, skip: bool = True):
        super(MixedGroupConv2d, self).__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels * groupsize(group)
        self.in_channels = in_channels * groupsize(group)
        self.alphas = torch.nn.Parameter(torch.zeros(np.prod([g+1 for g in group])+1), requires_grad=True)
        if discrete:
            self.alphas.data[:-2] = -np.inf
        elif prior: 
            self.alphas.data[:-2] = -2
        if baseline:
            self.alphas.data[:] = -np.inf
            if prior:
                self.alphas.data[-2] = 0
            else:
                self.alphas.data[0] = 0
        if skip:
            self.alphas.data[-1] = 0
        else:
            self.alphas.data[-1] = -np.inf
        self.norm = norm
        self.norms = torch.nn.Parameter(torch.zeros(np.prod([g+1 for g in group])+1), requires_grad=False)
        self.weights = torch.nn.ParameterList()
        if bias:
            self.bias = torch.nn.ParameterList()
        else:
            self.bias = None
        self.groups = []
        self.inchannelorders = []
        self.inchannelapply = []
        self.outchannelorders = []
        for i in range(group[0]+1):
            for j in range(group[1]+1):
                g = (i,j)
                in_c = int(self.in_channels/groupsize(g))
                out_c = int(self.out_channels/groupsize(g))
                weights = torch.nn.Parameter(torch.normal(mean = 0.0, std = 1 / (out_c * in_c)**(1/2), 
                    size=(out_c, in_c, groupsize(g), kernel_size, kernel_size)), requires_grad=True)
                self.norms.data[len(self.weights)] = torch.linalg.norm(weights)
                self.weights.append(weights)
                self.groups.append(g)
                if not discrete and self.group == (0,2) and g != (0,2):
                    self.outchannelorders.append(sum([[4*c,4*c+2,4*c+1,4*c+3] for c in range(int(self.out_channels/4))], start=[]))
                    self.inchannelapply.append([[2*c+1 for c in range(int(self.out_channels/2))],])
                    self.inchannelorders.append([sum([[4*c+3,4*c+2,4*c+1,4*c] for c in range(int(self.in_channels/4))], start=[]),])
                else:
                    self.outchannelorders.append(list(range(self.out_channels)))
                    self.inchannelapply.append([])
                    self.inchannelorders.append([])
                if bias:
                    self.bias.append(torch.nn.Parameter(torch.zeros(out_c), requires_grad=True))
        skip_weights = torch.zeros(size=(0,))
        if self.out_channels % self.in_channels != 0:
            self.alphas.data[-1] = -np.inf
        skip_weights = torch.nn.Parameter(skip_weights, requires_grad=True)
        self.weights.append(skip_weights)
        if bias:
            self.bias.append(torch.nn.Parameter(torch.zeros(self.out_channels), requires_grad=True))
        self.groups.append((-1,-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        alphas = torch.softmax(self.alphas, dim=0)

        if self.out_channels % self.in_channels == 0 and alphas[-1] > 0:
            out = alphas[-1]*torch.tile(x, dims=(1,self.out_channels//self.in_channels,1,1))
        else:
            out = torch.zeros(x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]).to(x.device)

        for layer in range(len(self.groups)-1):
            if alphas[layer] > 0:
                if len(x.shape)>1 and self.norm:
                    weights = self.weights[layer]/torch.linalg.norm(self.weights[layer])*self.norms[layer]
                else:
                    weights = self.weights[layer]

                if self.groups[layer][0] == 1:
                    order = [(i,j) for j in range(subgroupsize(self.groups[layer], 0)) for i in range(subgroupsize(self.groups[layer], 1))]
                    _filter = torch.stack([rotateflipstack_n(weights, i, subgroupsize(self.groups[layer], 1), j, subgroupsize(self.groups[layer], 0)) for (i,j) in order], dim = -5)
                else:
                    _filter = torch.stack([rotatestack_n(weights, i, groupsize(self.groups[layer])) for i in range(subgroupsize(self.groups[layer], 1))], dim = -5)

                if self.bias is not None:
                    _bias = torch.stack([self.bias[layer] for _ in range(groupsize(self.groups[layer]))], dim = 1)
                    _bias = _bias.reshape(self.out_channels)
                else:
                    _bias = None

                _filter = _filter.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

                for (apply, order) in zip(self.inchannelapply[layer], self.inchannelorders[layer]):
                    _filter[apply] = _filter[apply][:,order]
                _filter = _filter[self.outchannelorders[layer]]

                if len(x.shape)<=1:
                    return _filter
                
                y = torch.conv2d(x, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=_bias)

                out += alphas[layer]*y
  
        return out
    
    def countparams(self):
        return sum([torch.numel(w) for a, w in zip(torch.softmax(self.alphas, dim=0),self.weights) if a > 0])
    
    def regularization_loss(self, L2=True, reg_conv=0.0, reg_group=0.0, reg_alphas=0.0):
        weightsum = 0.0
        alphas = torch.softmax(self.alphas, dim=0)
        coeffs = torch.ones_like(alphas)*reg_group
        coeffs[0] = reg_conv
        for i in range(len(alphas)-1):
            if alphas[i] > 0:
                if L2:
                    weightsum += coeffs[i]*self.weights.pow(2).sum()
                else:
                    weightsum += coeffs[i]*self.weights.abs().sum()
        if L2:
            weightsum += reg_alphas*self.alphas.pow(2).sum()
        else:
            weightsum += reg_alphas*self.alphas.abs().sum()
        return weightsum

class DEANASNet(torch.nn.Module):

    def __init__(self, alphalr = 1e-3, weightlr = 1e-3, baseline: bool = False, superspace: tuple = (1,2), basechannels: int = 16, 
                 stages: int = 2, stagedepth: int = 4, pools: int = 4, kernel: int = 5, indim: int = 1, outdim: int = 10, 
                 hidden: int = 64, prior: bool = True, discrete: bool = False, norm: bool = True, skip: bool = True, reg_conv:float = 0.0, 
                 reg_group: float = 0.0):
        
        super(DEANASNet, self).__init__()
        self.alphalr = alphalr
        self.weightlr = weightlr
        self.reg_conv = L2_conv
        self.reg_group = L2_group
        self.superspace = superspace
        self.basechannels = basechannels
        self.stages = stages
        self.stagedepth = stagedepth
        self.pools = pools
        self.kernel = kernel
        self.indim = indim
        self.hidden = hidden
        self.outdim = outdim
        self.prior = prior
        self.skip = skip
        self.norm = norm
        self.discrete = discrete
        self.channels = [basechannels*2**i for i in range(stages) for _ in range(stagedepth)]
        self.kernels = [kernel for _ in range(len(self.channels))]
        self.blocks = torch.nn.ModuleList([])
        mlc = MixedLiftingConv2d(baseline=baseline, in_channels=indim, out_channels=self.channels[0], group=self.superspace, kernel_size=self.kernels[0], padding=self.kernels[0]//2, prior=prior, discrete=discrete, norm=norm, skip=skip)
        self.groups = mlc.groups
        self.blocks.append(torch.nn.Sequential(
            mlc,
            torch.nn.BatchNorm2d(self.channels[0]*groupsize(self.superspace)),
            torch.nn.ReLU(inplace=True)
            ))
        for i in range(1,len(self.channels)):
            self.blocks.append(torch.nn.Sequential(
                MixedGroupConv2d(baseline=baseline, in_channels=self.channels[i-1], out_channels=self.channels[i], group=self.superspace, kernel_size=self.kernels[i], padding=self.kernels[i]//2, prior=prior, discrete=discrete, norm=norm, skip=skip),
                torch.nn.BatchNorm2d(self.channels[i]*groupsize(self.superspace)),
                torch.nn.ReLU(inplace=True)
                ))
            if i%(len(self.channels)//pools) == 0 and i != len(self.channels)-1 and (not self.indim>1 or i > 2):
                self.blocks[i].add_module(name="pool", module = torch.nn.AvgPool2d((3,3), (2,2), padding=(1,1)))
        self.blocks[-1].add_module(name="pool", module = torch.nn.AvgPool2d((4,4), (1,1), padding=(0,0)))
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Linear(self.channels[-1]*groupsize(self.superspace), hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ELU(inplace=True),
        ))
        self.blocks.append(torch.nn.Sequential(torch.nn.Linear(hidden, outdim)))
        self.loss_function = torch.nn.CrossEntropyLoss()
        if discrete:
            self.optimizer = torch.optim.SGD(self.all_params(), lr=weightlr)
            self.alphaopt = None
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=weightlr)
            self.alphaopt = torch.optim.Adam(self.alphas(), lr=alphalr)
        self.gs = [self.superspace for _ in range(len(self.channels))]
        self.score = -1
        self.uuid = uuid.uuid4()
        self.parent = None

    def forward(self, x: torch.Tensor):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == len(self.blocks)-3:
                x = x.reshape(x.shape[0], -1)
        return x

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters():
            if "alphas" not in name and "norms" not in name and not name.endswith(str(np.prod([g+1 for g in self.superspace]))):
                yield param
    
    def named_params(self, recurse: bool = True):
        for name, param in self.named_parameters():
            if "alphas" not in name and "norms" not in name and not name.endswith("."+str(np.prod([g+1 for g in self.superspace]))):
                yield name, param
                        
    def alphas(self, recurse: bool = True):
        for name, param in self.named_parameters():
            if "alphas" in name:
                yield param
    
    def all_params(self, recurse: bool = True):
        for name, param in self.named_parameters():
            if "norms" not in name and not name.endswith("."+str(np.prod([g+1 for g in self.superspace]))):
                yield param
    
    def regularization_loss(self, L2 == True):
        weightsum = 0.0
        for i in range(len(self.blocks)):
            for key in self.blocks[i]._modules.keys():
                if isinstance(self.blocks[i]._modules[key], MixedGroupConv2d) or isinstance(self.blocks[i]._modules[key], MixedLiftingConv2d):
                    count += self.blocks[i]._modules[key].regularization_loss(L2=L2, reg_conv=self.reg_conv, reg_group=self.reg_group)
        return count

    def countparams(self):
        count = 0
        for i in range(len(self.blocks)):
            for key in self.blocks[i]._modules.keys():
                if isinstance(self.blocks[i]._modules[key], MixedGroupConv2d) or isinstance(self.blocks[i]._modules[key], MixedLiftingConv2d):
                    count += self.blocks[i]._modules[key].countparams()
                else:
                    count += sum(p.numel() for p in self.blocks[i]._modules[key].parameters())
        return count

    def offspring(self, i, groupnew, verbose = False):
        assert all([sum(a > -np.inf) <= 2 for a in self.alphas()])
        offspring = DEANASNet(alphalr = self.alphalr, weightlr = self.weightlr, superspace = self.superspace, basechannels = self.basechannels, stages = self.stages, stagedepth = self.stagedepth, pools = self.pools, kernel = self.kernel, indim = self.indim, outdim = self.outdim, prior = self.prior, discrete=self.discrete)
        offspring.load_state_dict(self.state_dict())
        for j in range(len(self.channels)):
            offspring.blocks[j]._modules["0"].outchannelorders = copy.deepcopy(self.blocks[j]._modules["0"].outchannelorders)
            offspring.blocks[j]._modules["0"].inchannelapply = copy.deepcopy(self.blocks[j]._modules["0"].inchannelapply)
            offspring.blocks[j]._modules["0"].inchannelorders = copy.deepcopy(self.blocks[j]._modules["0"].inchannelorders)
        offspring.parent = self.uuid
        offspring.gs = [g for g in self.gs]
        if i < 0 or any([groupnew[j]>self.gs[i][j] for j in range(len(groupnew))]):
            return offspring
        offspring.gs[i] = groupnew            
        indold = np.argmax(list(offspring.alphas())[i].data[:-1])
        groupold = offspring.blocks[i]._modules["0"].groups[indold]
        for a in range(len(list(self.alphas()))):
            list(offspring.alphas())[a].data.copy_(list(self.alphas())[a].data)
        indnew = offspring.blocks[i]._modules["0"].groups.index(groupnew)
        if offspring.gs[i] == self.gs[i]:
            assert indold == indnew
            return offspring
        list(offspring.alphas())[i].data[indnew] = list(self.alphas())[i].data[indold]
        list(offspring.alphas())[i].data[indold] = list(self.alphas())[i].data[indnew]
        # if verbose:
        #     for a in range(offspring.blocks[i]._modules["0"].weights[indold].shape[0]):
        #         for b in range(offspring.blocks[i]._modules["0"].weights[indold].shape[1]):
        #             for c in range(offspring.blocks[i]._modules["0"].weights[indold].shape[2]):
        #                 for d in range(offspring.blocks[i]._modules["0"].weights[indold].shape[3]):
        #                     if len(offspring.blocks[i]._modules["0"].weights[indold].shape) > 4:
        #                         for e in range(offspring.blocks[i]._modules["0"].weights[indold].shape[4]):
        #                             self.blocks[i]._modules["0"].weights[indold].data[a,b,c,d,e] = (a*10**4+b*10**3+c*10**2)*1+d*10+e
        #                             offspring.blocks[i]._modules["0"].weights[indold].data[a,b,c,d,e] = (a*10**4+b*10**3+c*10**2)*1+d*10+e
        #                     else:
        #                         self.blocks[i]._modules["0"].weights[indold].data[a,b,c,d] = (a*10**4+b*10**3+c*10**2)//10+d
        #                         offspring.blocks[i]._modules["0"].weights[indold].data[a,b,c,d] = (a*10**4+b*10**3+c*10**2)//10+d
        weights = self.blocks[i]._modules["0"].weights[indold]
        if verbose:
            print(i, indold, indnew, groupold, groupnew)
        if groupold[0] == 1:
            order = [(f,r) for f in range(subgroupsize(groupold, 0)//subgroupsize(groupnew, 0)) for r in range(subgroupsize(groupold, 1)//subgroupsize(groupnew, 1))]
            if i == 0:
                weightmats = [rotateflip_n(weights, r, subgroupsize(groupold, 1), f, subgroupsize(groupold, 0)) for (f,r) in order]
            else:
                weightmats = [rotateflipstack_n(weights, r, subgroupsize(groupold, 1), f, subgroupsize(groupold, 0)) for (f,r) in order]
        else:
            if i == 0:
                weightmats = [rotate_n(weights, r, groupsize(groupold)) for r in range(subgroupsize(groupold, 1)//subgroupsize(groupnew, 1))]
            else:
                weightmats = [rotatestack_n(weights, r, groupsize(groupold)) for r in range(subgroupsize(groupold, 1)//subgroupsize(groupnew, 1))]
        semi_filter = torch.stack(weightmats, dim = -5)
        if self.blocks[i]._modules["0"].bias is not None:
            bias = torch.stack([self.blocks[i]._modules["0"].bias[indold] for _ in range(groupdifference(groupold, groupnew))], dim = 1).reshape(-1)
            offspring.blocks[i]._modules["0"].bias[indnew] = torch.nn.Parameter(bias)
        if verbose and i > 0:
            print(semi_filter[0,1,0,:,0,1])
            print(weights.shape, offspring.blocks[i]._modules["0"].weights[indnew].shape, semi_filter.shape)
        semi_filter = semi_filter.reshape(offspring.blocks[i]._modules["0"].weights[indnew].shape)
        offspring.blocks[i]._modules["0"].weights[indnew] = torch.nn.Parameter(semi_filter)
        offspring.blocks[i]._modules["0"].norms[indnew] = torch.linalg.norm(offspring.blocks[i]._modules["0"].weights[indnew].data)
        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = self.blocks[i]._modules["0"].outchannelorders[indold]
        if i != 0:
            offspring.blocks[i]._modules["0"].inchannelapply[indnew] = self.blocks[i]._modules["0"].inchannelapply[indold]
            offspring.blocks[i]._modules["0"].inchannelorders[indnew] = self.blocks[i]._modules["0"].inchannelorders[indold]
        inchannels = offspring.blocks[i]._modules["0"].in_channels
        outchannels = offspring.blocks[i]._modules["0"].out_channels
        if groupold[1] == 2 and groupnew == (0,1) and i!=0:
            offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[4*c,4*c+2,4*c+1,4*c+3] for c in range(int(outchannels/4))], start=[])
            offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [[2*c+1 for c in range(int(outchannels/2))],]
            offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[4*c+3,4*c+2,4*c+1,4*c] for c in range(int(inchannels/4))], start=[]),]
        elif groupold == (1,2):
            if i != 0:
                offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [[2*c+1 for c in range(int(outchannels/2))],]
            if groupnew == (1,0):
                if i != 0:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+2,8*c+4,8*c+6,8*c+1,8*c+7,8*c+5,8*c+3] for c in range(int(outchannels/8))], start=[])
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[8*c+5,8*c+6,8*c+7,8*c+4,8*c+1,8*c+2,8*c+3,8*c] for c in range(int(inchannels/8))], start=[]),]
                else:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+1,8*c+2,8*c+3,8*c+6,8*c+5,8*c+4,8*c+7] for c in range(int(outchannels/8))], start=[])
            elif groupnew == (1,1):
                if i != 0:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+4,8*c+1,8*c+5,8*c+2,8*c+7,8*c+3,8*c+6] for c in range(int(outchannels/8))], start=[])
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [sum([[8*c+2,8*c+3,8*c+6,8*c+7] for c in range(int(outchannels/8))], start=[]),]
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[8*c+5,8*c+6,8*c+7,8*c+4,8*c+1,8*c+2,8*c+3,8*c] for c in range(int(inchannels/8))], start=[]),]
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew].append(sum([[8*c+1,8*c+5] for c in range(int(outchannels/8))], start=[]))
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+3,8*c+2,8*c+1,8*c,8*c+7,8*c+6,8*c+5,8*c+4] for c in range(int(inchannels/8))], start=[]))
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew].append(sum([[8*c+2,8*c+6] for c in range(int(outchannels/8))], start=[]))
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+1,8*c,8*c+3,8*c+2,8*c+5,8*c+4,8*c+7,8*c+6] for c in range(int(inchannels/8))], start=[]))
                else:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+1,8*c+2,8*c+3,8*c+4,8*c+7,8*c+6,8*c+5] for c in range(int(outchannels/8))], start=[])
            elif groupnew == (0,1):
                offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+1,8*c+4,8*c+5,8*c+2,8*c+3,8*c+6,8*c+7] for c in range(int(outchannels/8))], start=[])
            elif groupnew == (0,2) and i == 0:
                offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+2,8*c+4,8*c+6,8*c+1,8*c+3,8*c+5,8*c+7] for c in range(int(outchannels/8))], start=[])
        elif groupold == (1,1):
            if groupnew == (1,0):
                if i != 0:
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [[4*c+1 for c in range(int(outchannels/4))],]
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[8*c+5,8*c+6,8*c+7,8*c+4,8*c+1,8*c+2,8*c+3,8*c] for c in range(int(inchannels/8))], start=[]),]
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew].append([4*c+2 for c in range(int(outchannels/4))])
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+3,8*c+2,8*c+1,8*c,8*c+7,8*c+6,8*c+5,8*c+4] for c in range(int(inchannels/8))], start=[]))
                    offspring.blocks[i]._modules["0"].inchannelapply[indnew].append([4*c+3 for c in range(int(outchannels/4))])
                    offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+6,8*c+5,8*c+4,8*c+7,8*c+2,8*c+1,8*c+0,8*c+3] for c in range(int(inchannels/8))], start=[]))
                if offspring.superspace == (1,2):
                    if i != 0:
                        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+4,8*c+2,8*c+6,8*c+1,8*c+7,8*c+3,8*c+5] for c in range(int(outchannels/8))], start=[])
                    else:
                        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+1,8*c+2,8*c+3,8*c+6,8*c+5,8*c+4,8*c+7] for c in range(int(outchannels/8))], start=[])
                else:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[4*c,4*c+2,4*c+1,4*c+3] for c in range(int(outchannels/4))], start=[])
            elif groupnew == (0,1):
                if offspring.superspace == (1,2):
                    if i != 0:
                        offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [[4*c+1 for c in range(int(outchannels/4))],]
                        offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[4*c+3,4*c+2,4*c+1,4*c] for c in range(int(inchannels/4))], start=[]),]
                        offspring.blocks[i]._modules["0"].inchannelapply[indnew].append([4*c+2 for c in range(int(outchannels/4))])
                        offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+6,8*c+5,8*c+4,8*c+7,8*c+2,8*c+1,8*c,8*c+3] for c in range(int(inchannels/8))], start=[]))
                        offspring.blocks[i]._modules["0"].inchannelapply[indnew].append([4*c+3 for c in range(int(outchannels/4))])
                        offspring.blocks[i]._modules["0"].inchannelorders[indnew].append(sum([[8*c+5,8*c+6,8*c+7,8*c+4,8*c+1,8*c+2,8*c+3,8*c] for c in range(int(inchannels/8))], start=[]))
                        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+4,8*c+1,8*c+5,8*c+2,8*c+7,8*c+3,8*c+6] for c in range(int(outchannels/8))], start=[])
                    else:
                        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[8*c,8*c+1,8*c+4,8*c+5,8*c+2,8*c+7,8*c+6,8*c+3] for c in range(int(outchannels/8))], start=[])
                else:
                    offspring.blocks[i]._modules["0"].outchannelorders[indnew] = sum([[4*c,4*c+2,4*c+1,4*c+3] for c in range(int(outchannels/4))], start=[])
                    if i != 0:
                        offspring.blocks[i]._modules["0"].inchannelapply[indnew] = [sum([[8*c+7,8*c+4,8*c+5,8*c+6] for c in range(int(outchannels/8))], start=[]),]
                        offspring.blocks[i]._modules["0"].inchannelorders[indnew] = [sum([[8*c+5,8*c+6,8*c+7,8*c+4,8*c+1,8*c+2,8*c+3,8*c] for c in range(int(inchannels/8))], start=[]),]
        offspring.blocks[i]._modules["0"].outchannelorders[indnew] = [oco for oco in offspring.blocks[i]._modules["0"].outchannelorders[indnew] if oco < offspring.blocks[i]._modules["0"].out_channels]
        if i != 0:
            for j in range(len(offspring.blocks[i]._modules["0"].inchannelapply[indnew])):
                offspring.blocks[i]._modules["0"].inchannelapply[indnew][j] = [ica for ica in offspring.blocks[i]._modules["0"].inchannelapply[indnew][j] if ica < offspring.blocks[i]._modules["0"].out_channels]
            for j in range(len(offspring.blocks[i]._modules["0"].inchannelorders[indnew])):
                offspring.blocks[i]._modules["0"].inchannelorders[indnew][j] = [ico for ico in offspring.blocks[i]._modules["0"].inchannelorders[indnew][j] if ico < offspring.blocks[i]._modules["0"].in_channels]
        if verbose:
            if i != 0:
                print(offspring.blocks[i]._modules["0"](torch.Tensor([]))[0:8,0:6,0,1])
                print(self.blocks[i]._modules["0"](torch.Tensor([]))[0:8,0:6,0,1])
            else:
                print("new weights", offspring.blocks[i]._modules["0"].weights[indnew])
                print("old weights", self.blocks[i]._modules["0"].weights[indold])

                print("child _filter:", offspring.blocks[i]._modules["0"](torch.Tensor([])).shape, offspring.blocks[i]._modules["0"](torch.Tensor([]))[:,0,:,1])
                print("parent _filter:", self.blocks[i]._modules["0"](torch.Tensor([])).shape, self.blocks[i]._modules["0"](torch.Tensor([]))[:,0,:,1])
            if i < len(self.channels)-1:
                print("next")
                print(offspring.blocks[i+1]._modules["0"](torch.Tensor([]))[0:10,0:6,0,1])
                print(self.blocks[i+1]._modules["0"](torch.Tensor([]))[0:10,0:6,0,1])
            if not torch.allclose(offspring.blocks[i]._modules["0"](torch.Tensor([])), self.blocks[i]._modules["0"](torch.Tensor([]))):
                print("not close")
        return offspring
    
    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(len(self.gs[0])):
            for i in range(1, self.gs[-1][d]+1):
                g = list(self.gs[-1])
                g[d] -= i
                child = self.offspring(len(self.gs)-1, tuple(g))
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                child = self.offspring(i-1, self.gs[i])
                if all([child.gs != sibling.gs for sibling in candidates]):
                    candidates.append(child)
        return candidates

