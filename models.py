import torch
import torch.nn.functional as F
import numpy as np
import copy
import uuid
import escnn
from escnn import gspaces


def rotmat2d(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])
def rotmat3d(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[1, 0, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta), 0],
                         [0, torch.sin(theta), torch.cos(theta), 0]])

def rot(x, theta, dtype): #TODO test and fix
    rot_mat = rotmat3d(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def rotate_4(x: torch.Tensor, r: int) -> torch.Tensor:
    return x.rot90(r, dims=(-2, -1))

def rotate_n(x: torch.Tensor, r: int, n: int) -> torch.Tensor:
    if r == 0:
        roty = x
    elif n == 2:
        roty = rotate_4(x, 2*r)
    elif n == 4:
        roty = rotate_4(x, r)
    else:
        roty = rot(x, 2*r*np.pi/n, type(x))
    return roty

def rotatestack_n(y: torch.Tensor, r: int, n: int, order = None) -> torch.Tensor:
    # `y` is a function over pn, i.e. over the pixel positions and over the elements of the group C_n.
    # This method implements the action of a rotation `r` on `y`, assuming that the last two dimensions 
    # (`dim=-2` and `dim=-1`) of `y` are the spatial dimensions while `dim=-3` has size `n` and is the 
    # C_n dimension in group order. All other dimensions are considered batch dimensions
    if order is None:
        order = list(range(n))
    assert len(y.shape) >= 3
    assert y.shape[-3] == n
    assert len(order) == n

    roty = rotate_n(y, r, n) #first rotate y
    #print(r, n, [(n-r+i)%n for i in order])
    roty = torch.stack([torch.select(roty, -3, (n-r+i)%n) for i in order], dim=-3) #then reorder the group elements
    return roty

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
    #TODO: swap dims?
    weight = torch.cat([rotate_n(torch.cat([parentweight[:,:,order[i]] for i in range(len(order))], dim=0), j, groupsize(parentg)) for j in range(groupdifference(parentg, childg))], dim=1)[outchannelorder]
    #weight = torch.cat([rotate_n(torch.cat([parentweight[:,:,order[i]] for i in range(len(order))], dim=0), 0, groupsize(parentg)) for j in range(groupdifference(parentg, childg))], dim=1)
    #weight = parentweight.repeat((groupdifference(parentg, childg),1,1,1,1)).reshape(outchannels, inchannels, -1, parentweight.shape[3], parentweight.shape[4])
    #print(weight.shape)
    if childg == 0:
        weight = torch.unsqueeze(weight, dim = 2)
    return weight[:, inchannelorder]

def replicate(parent: torch.nn.Module, childg, inchannels, outchannels):

    filter, bias = parent.build_filter()
    #reorder?
    filter = filter.clone().reshape(parent.out_channels * groupsize(parent.group), parent.in_channels * groupsize(parent.group), parent.kernel_size, parent.kernel_size)
    filter = filter.reshape(outchannels, groupsize(childg), inchannels * childg, parent.kernel_size, parent.kernel_size)

    child = parent.initchild(childg, inchannels, outchannels)

    child.weight = filter[:, 0]
    if bias is not None:
        bias = bias.clone().reshape(parent.out_channels * groupsize(parent.group)).reshape(outchannels, groupsize(childg))
        child.bias = bias[:, 0]

    return child

def groupsize(g: tuple):
    return 2**sum(g)

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

    def replicate(self, group:tuple, out_channels: int):
        filter, bias = self.build_filter()
        #reorder?
        filter = filter.clone().reshape(self.out_channels * groupsize(self.group), self.in_channels, self.kernel_size, self.kernel_size)
        filter = filter.reshape(out_channels, groupsize(group), self.in_channels, self.kernel_size, self.kernel_size)

        child = LiftingConv2d(group, self.in_channels, out_channels, self.kernel_size, self.padding, self.bias is not None)

        child.weight.data = filter[:, 0]
        if bias is not None:
            bias = bias.clone().reshape(self.out_channels * groupsize(self.group)).reshape(out_channels, groupsize(group))
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

        _filter = torch.stack([rotate_n(self.weight.data, i, groupsize(self.group))
                            for i in self.order], dim=-4)

        if self.bias is not None:
            _bias = torch.stack([self.bias.data for _ in range(groupsize(self.group))], dim=1)
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

    def __init__(self, group: tuple, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):
        
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
    
    def replicate(self, group:tuple, in_channels: int, out_channels: int):
        filter, bias = self.build_filter()
        #reorder?
        filter = filter.clone().reshape(self.out_channels * groupsize(self.group), self.in_channels * groupsize(self.group), self.kernel_size, self.kernel_size)
        filter = filter.reshape(out_channels, groupsize(group), in_channels, groupsize(group), self.kernel_size, self.kernel_size)

        child = GroupConv2d(group, in_channels, out_channels, self.kernel_size, self.padding, self.bias is not None)
        child.weight.data = filter[:, 0]
        if bias is not None:
            bias = bias.clone().reshape(self.out_channels * groupsize(self.group)).reshape(out_channels, groupsize(group))
            child.bias.data = bias[:, 0]

        return child
  
    def build_filter(self) -> torch.Tensor:
        
        #print("weight", self.weight.shape)

        _filter = torch.stack([rotatestack_n(self.weight.data, i, groupsize(self.group)) for i in range(groupsize(self.group))], dim = -5)

        #print("filter", _filter.shape)

        if self.bias is not None:
            _bias = torch.stack([self.bias.data for _ in range(groupsize(self.group))], dim = 1)
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

        x = x.view(x.shape[0], self.in_channels*groupsize(self.group), x.shape[-2], x.shape[-1])

        out = torch.conv2d(x, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=_bias)

        return out.view(-1, self.out_channels, groupsize(self.group), out.shape[-2], out.shape[-1])

class Reshaper(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, in_groupsize: int, out_groupsize: int):

        super(Reshaper, self).__init__()

        assert in_groupsize*in_channels == out_groupsize*out_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_groupsize = in_groupsize
        self.out_groupsize = out_groupsize
        #self.in_order = [int('{:0{width}b}'.format(n, width=int(np.log2(in_groupsize)))[::-1], 2) for n in range(in_groupsize)]
        #self.out_order = [int('{:0{width}b}'.format(n, width=int(np.log2(out_groupsize)))[::-1], 2) for n in range(out_groupsize)]
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
    
    def __init__(self, gs = [(0,0) for _ in range(6)], parent = None):
        
        super(TDRegEquiCNN, self).__init__()

        self.uuid = uuid.uuid4()
        if parent is not None:
            self.parent = parent.uuid
        else:
            self.parent = None
        self.gs = gs
        self.channels = [24, 48, 48, 96, 96, 64]
        #self.channels = [64, 128, 128, 256, 256, 64]
        self.kernels = [7, 5, 5, 5, 5, 5]
        self.paddings = [1, 2, 2, 2, 2, 1]
        self.blocks = torch.nn.ModuleList([])
        self.architect(parent)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.SGD(self.parameters(), lr=5e-5)
        #for (n,p) in self.named_parameters():
            #if p.requires_grad:
                #print(n, p.shape)

    def architect(self, parent = None):
        #TODO test morphism
        init = (parent is None)
        if not init and self.gs == parent.gs:
            self.blocks = copy.deepcopy(parent.blocks)
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            #self = copy.deepcopy(parent)
            return 
        self.blocks.append(torch.nn.Sequential(
                LiftingConv2d(self.gs[0], 1, int(self.channels[0]/groupsize(self.gs[0])), self.kernels[0], self.paddings[0], bias=True),
                torch.nn.BatchNorm3d(int(self.channels[0]/groupsize(self.gs[0]))),
                torch.nn.ReLU(inplace=True)
            )
        )
        if not init:
            self.blocks[0]._modules["0"] = parent.blocks[0]._modules["0"].replicate(self.gs[0], int(self.channels[0]/groupsize(self.gs[0])))
            # parentweight = parent.blocks[0]._modules["0"].weight.data
            # self.blocks[0]._modules["0"].weight.data = torch.cat([rotate_n(parentweight.clone(), i, groupsize(parent.gs[0])) for i in range(groupdifference(parent.gs[0], self.gs[0]))], dim=0)
            # if self.gs[0] == parent.gs[0]:
            #     self.blocks[0]._modules["0"].bias.data = parent.blocks[0]._modules["0"].bias.data.clone()
            # #TODO: else
        
        for i in range(1, len(self.gs)):
            #print(i, self.gs[i], int(self.channels[i-1]/groupsize(self.gs[i])), int(self.channels[i]/groupsize(self.gs[i])))
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
                self.blocks[i]._modules["0"] = parent.blocks[i]._modules["0"].replicate(self.gs[i], int(self.channels[i-1]/groupsize(self.gs[i])), int(self.channels[i]/groupsize(self.gs[i])))
                # weight = adapt(parent.blocks[i]._modules["0"].weight.data.clone(), parent.gs[i], self.gs[i], 
                #                self.blocks[i]._modules["0"].weight.shape[1], self.blocks[i]._modules["0"].weight.shape[0])
                # self.blocks[i]._modules["0"].weight.data = weight
                # #TODO: repeat_interleave?
                # self.blocks[i]._modules["0"].bias.data = parent.blocks[i]._modules["0"].bias.data.repeat(groupdifference(parent.gs[i], self.gs[i]))

            if i < len(self.gs)-1:
                if self.gs[i+1] != self.gs[i]:
                    self.blocks[i].add_module(name="reshaper", module = Reshaper(in_channels=int(self.channels[i]/groupsize(self.gs[i])), out_channels=int(self.channels[i]/groupsize(self.gs[i+1])), in_groupsize=groupsize(self.gs[i]), out_groupsize=groupsize(self.gs[i+1])))

        if init:
            self.blocks.append(torch.nn.AvgPool3d((groupsize(self.gs[-1]),5,5), (1,1,1), padding=(0,0,0)))
            #print(int(self.channels[-1]/groupsize(self.gs[-1])))
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
                self.full1 = torch.nn.Sequential(
                    torch.nn.Linear(int(self.channels[-1]/groupsize(self.gs[-1])), 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ELU(inplace=True),
                )
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
        for d in range(1,len(self.gs[0])):
            #if self.gs[-1][d] >= 0:
            for i in range(1, self.gs[-1][d]+1):
                g = list(self.gs[0])
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
        child = TDRegEquiCNN(gs = gs, parent=self)
        # if i < 0:
        #     for i in range(len(self.blocks)):
        #         for key in child.blocks[i]._modules:
        #             if hasattr(child.blocks[i]._modules[key], "weight"):
        #                 print(i, key, torch.allclose(child.blocks[i]._modules[key].weight, self.blocks[i]._modules[key].weight, atol=1e-5, rtol=1e-5))
        #     for key in child.full1._modules:
        #         if hasattr(child.full1._modules[key], "weight"):
        #             print("full1", key, torch.allclose(child.full1._modules[key].weight, self.full1._modules[key].weight, atol=1e-5, rtol=1e-5))
        #     print("full2", key, torch.allclose(child.full2.weight, self.full2.weight, atol=1e-5, rtol=1e-5))
        return child


def first(od):
    for item in od.values():
        return item

def sgid(index, previous_index = None):
    if previous_index is not None:
        if previous_index[0] == 0 and index[0] == 0:
            return 2**index[1]
        if previous_index[1] == 0 and index[1] == 0:
            return index[0]
    if index[0] == 0:
        return (None, 2**index[1])
    return (0, 2**index[1])
    

class EquiCNN(torch.nn.Module):
    
    def __init__(self, reset = False, gs = [(0,0) for _ in range(6)], parent = None):
        
        super(EquiCNN, self).__init__()

        self.uuid = uuid.uuid4()
        if parent is not None:
            self.parent = parent.uuid
        else:
            self.parent = None
        self.superspace = gspaces.flipRot2dOnR2(N=8)
        self.gspaces = np.transpose(np.column_stack(([gspaces.trivialOnR2(), gspaces.rot2dOnR2(N=2), gspaces.rot2dOnR2(N=4), gspaces.rot2dOnR2(N=8)], [gspaces.flip2dOnR2(), gspaces.flipRot2dOnR2(N=2), gspaces.flipRot2dOnR2(N=4), gspaces.flipRot2dOnR2(N=8)])))
        self.gs = gs
        self.channels = [24, 48, 48, 96, 96, 64]
        self.kernels = [7, 5, 5, 5, 5, 5]
        self.paddings = [1, 2, 2, 2, 2, 1]
        self.blocks = torch.nn.ModuleList([])
        self.architect(parent)
        self.reset = reset #TODO: if true, reinit changed layer, else network morphism 
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

    def architect(self, parent = None):
        init = (parent is None)
        if not init and self.gs == parent.gs:
            self.input_type = parent.input_type
            self.blocks = torch.nn.ModuleList([escnn.nn.SequentialModule(block._modules) if hasattr(block, "_modules") and len(block._modules) > 0 else copy.deepcopy(block) for block in parent.blocks])
            self.superspace = parent.superspace
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            for i in range(len(self.gs)):
                self.blocks[i]._modules['0'].weights = copy.deepcopy(parent.blocks[i]._modules['0'].weights)
            return
        G, _, _ = self.superspace.restrict(sgid(self.gs[0])) 
        in_type = escnn.nn.FieldType(G, [G.trivial_repr])
        self.input_type = in_type
        for i in range(len(self.gs)):
            out_type = escnn.nn.FieldType(G, int(self.channels[i]/np.sqrt(G.fibergroup.order()))*[G.regular_repr])
            #if init or self.gs[i] != parent.gs[i]:
            #if init or (in_type != parent.blocks[i].in_type or out_type != first(parent.blocks[i]._modules).out_type):
            self.blocks.append(escnn.nn.SequentialModule(
                escnn.nn.R2Conv(in_type, out_type, kernel_size=self.kernels[i], padding=self.paddings[i], bias=False),
                escnn.nn.InnerBatchNorm(out_type),
                escnn.nn.ReLU(out_type, inplace=True)
            ))
            if i == 1 or i == 3:
                self.blocks[i].add_module(name="pool", module=escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))                    
            if not (init or (in_type != parent.blocks[i].in_type or out_type != first(parent.blocks[i]._modules).out_type)):
                self.blocks[i]._modules['0'].weights = copy.deepcopy(parent.blocks[i]._modules['0'].weights)


            if i < len(self.gs)-1:
                if self.gs[i] != self.gs[i+1] and "restrict" not in self.blocks[i]._modules.keys():
                    sg = sgid(self.gs[i+1], self.gs[i])
                    restrict = escnn.nn.RestrictionModule(self.blocks[i].out_type, sg)
                    self.blocks[i].add_module(name="restrict", module=restrict)
                    G, _, _ = copy.deepcopy(G).restrict(sg)
                    disentangle = escnn.nn.DisentangleModule(restrict.out_type)
                    self.blocks[i].add_module(name="disentangle", module=disentangle)
                    out_type = disentangle.out_type
                if self.gs[i] == self.gs[i+1] and "restrict" in self.blocks[i]._modules.keys():
                    del self.blocks[i]._modules["restrict"]
                    del self.blocks[i]._modules["disentangle"]
            in_type = out_type
        if init:
            self.blocks.append(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0))
            self.blocks.append(escnn.nn.GroupPooling(out_type))
            self.full1 = torch.nn.Sequential(
                torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
            )
            self.full2 = torch.nn.Linear(64, 10)
        else:
            self.blocks += [copy.deepcopy(block) for block in parent.blocks[-2:]]
        
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            if out_type != parent.blocks[len(self.gs)].out_type:
                self.blocks[len(self.gs)] = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
                self.blocks[len(self.gs)+1] = escnn.nn.GroupPooling(out_type)
                self.full1 = torch.nn.Sequential(
                    torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ELU(inplace=True),
                )

    def forward(self, input: torch.Tensor):
        x = escnn.nn.GeometricTensor(input, self.input_type)
        for (i,block) in enumerate(self.blocks):
            x = block(x)
        x = x.tensor
        x = self.full1(x.reshape(x.shape[0], -1))
        return self.full2(x)

    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(1,len(self.gs[0])):
            if self.gs[0][d] < self.gspaces.shape[d]-1:
                g = list(self.gs[0])
                g[d] += 1
                candidates.append(self.offspring(0, tuple(g)))
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                candidates.append(self.offspring(i, self.gs[i-1]))
        return candidates

    def offspring(self, i, G):
        gs = [g for g in self.gs]
        if i >= 0:
            gs[i] = G
        child = EquiCNN(reset = self.reset, gs = gs, parent=self)
        return child



class TDEquiCNN(torch.nn.Module):
    
    def __init__(self, reset = False, gs = [(0,0) for _ in range(6)], parent = None):
        
        super(TDEquiCNN, self).__init__()

        self.uuid = uuid.uuid4()
        if parent is not None:
            self.parent = parent.uuid
        else:
            self.parent = None
        self.superspace = gspaces.flipRot2dOnR2(N=8)
        self.gspaces = np.transpose(np.column_stack(([gspaces.trivialOnR2(), gspaces.rot2dOnR2(N=2), gspaces.rot2dOnR2(N=4), gspaces.rot2dOnR2(N=8)], [gspaces.flip2dOnR2(), gspaces.flipRot2dOnR2(N=2), gspaces.flipRot2dOnR2(N=4), gspaces.flipRot2dOnR2(N=8)])))
        self.gs = gs
        #self.channels = [24, 48, 48, 96, 96, 64]
        self.channels = [8, 8, 8, 8, 8, 64]
        self.kernels = [7, 5, 5, 5, 5, 5]
        self.paddings = [1, 2, 2, 2, 2, 1]
        self.blocks = torch.nn.ModuleList([])
        self.architect(parent)
        self.reset = reset #TODO: if true, reinit changed layer, else network morphism 
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

    def architect(self, parent = None):
        init = (parent is None)
        if not init and self.gs == parent.gs:
            self.input_type = parent.input_type
            self.blocks = torch.nn.ModuleList([escnn.nn.SequentialModule(block._modules) if hasattr(block, "_modules") and len(block._modules) > 0 else copy.deepcopy(block) for block in parent.blocks])
            self.superspace = parent.superspace
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            for i in range(len(self.gs)):
                self.blocks[i]._modules['0'].weights = copy.deepcopy(parent.blocks[i]._modules['0'].weights)
            return
        G, _, _ = self.superspace.restrict(sgid(self.gs[0])) 
        in_type = escnn.nn.FieldType(G, [G.trivial_repr])
        self.input_type = in_type
        for i in range(len(self.gs)):
            out_type = escnn.nn.FieldType(G, int(self.channels[i])*[G.regular_repr]) #/np.sqrt(G.fibergroup.order())
            #if init or self.gs[i] != parent.gs[i]:
            #if init or (in_type != parent.blocks[i].in_type or out_type != first(parent.blocks[i]._modules).out_type):
            self.blocks.append(escnn.nn.SequentialModule(
                escnn.nn.R2Conv(in_type, out_type, kernel_size=self.kernels[i], padding=self.paddings[i], bias=False),
                escnn.nn.InnerBatchNorm(out_type),
                escnn.nn.ReLU(out_type, inplace=True)
            ))
            if i == 1 or i == 3:
                self.blocks[i].add_module(name="pool", module=escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))                    
            if not (init or (in_type != parent.blocks[i].in_type or out_type != first(parent.blocks[i]._modules).out_type)):
                self.blocks[i]._modules['0'].weights = copy.deepcopy(parent.blocks[i]._modules['0'].weights)


            if i < len(self.gs)-1:
                if self.gs[i] != self.gs[i+1] and "restrict" not in self.blocks[i]._modules.keys():
                    sg = sgid(self.gs[i+1], self.gs[i])
                    restrict = escnn.nn.RestrictionModule(self.blocks[i].out_type, sg)
                    self.blocks[i].add_module(name="restrict", module=restrict)
                    G, _, _ = copy.deepcopy(G).restrict(sg)
                    disentangle = escnn.nn.DisentangleModule(restrict.out_type)
                    self.blocks[i].add_module(name="disentangle", module=disentangle)
                    out_type = disentangle.out_type
                if self.gs[i] == self.gs[i+1] and "restrict" in self.blocks[i]._modules.keys():
                    del self.blocks[i]._modules["restrict"]
                    del self.blocks[i]._modules["disentangle"]
            in_type = out_type
        if init:
            self.blocks.append(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0))
            self.blocks.append(escnn.nn.GroupPooling(out_type))
            self.full1 = torch.nn.Sequential(
                torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
            )
            self.full2 = torch.nn.Linear(64, 10)
        else:
            self.blocks += [copy.deepcopy(block) for block in parent.blocks[-2:]]
        
            self.full1 = copy.deepcopy(parent.full1)
            self.full2 = copy.deepcopy(parent.full2)
            if out_type != parent.blocks[len(self.gs)].out_type:
                self.blocks[len(self.gs)] = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
                self.blocks[len(self.gs)+1] = escnn.nn.GroupPooling(out_type)
                self.full1 = torch.nn.Sequential(
                    torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ELU(inplace=True),
                )

    def forward(self, input: torch.Tensor):
        x = escnn.nn.GeometricTensor(input, self.input_type)
        for (i,block) in enumerate(self.blocks):
            x = block(x)
        x = x.tensor
        x = self.full1(x.reshape(x.shape[0], -1))
        return self.full2(x)

    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(1,len(self.gs[0])):
            if self.gs[0][d] < self.gspaces.shape[d]-1:
                g = list(self.gs[0])
                g[d] += 1
                candidates.append(self.offspring(0, tuple(g)))
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                candidates.append(self.offspring(i, self.gs[i-1]))
        return candidates

    def offspring(self, i, G):
        gs = [g for g in self.gs]
        if i >= 0:
            gs[i] = G
        child = TDEquiCNN(reset = self.reset, gs = gs, parent=self)
        return child
