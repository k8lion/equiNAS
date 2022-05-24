import torch
import escnn
from escnn import gspaces
import numpy as np


class UnsteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, width_equated = True):
        
        super(UnsteerableCNN, self).__init__()    

        if width_equated:
            widths = [1, 192, 384, 768, 512]
        else:
            widths = [1, 48, 96, 192, 64]

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[0], widths[1], kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(widths[1]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[1], widths[2], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[2]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[2], widths[2], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[2]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[2], widths[3], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[3]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[3], widths[3], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[3]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[3], widths[4], kernel_size=5, padding=1, bias=False),
            torch.nn.BatchNorm2d(widths[4]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool3 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.gpool = torch.nn.AdaptiveAvgPool2d((1,1))
        
        self.fully_net1 = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
        )

        self.fully_net2 = torch.nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor):
        self.K = np.zeros((x.size(0), x.size(0)))

        for block in [self.block1, self.block2, self.pool1, self.block3, self.block4, self.pool2, self.block5, self.block6, self.pool3, self.gpool]:
            x = block(x)
            with torch.no_grad():
                x_ = x.view(x.size(0), -1)
                x_ = (x_ > 0).float()
                K = x_ @ x_.t()
                K2 = (1.-x_) @ (1.-x_.t())
                self.K += K.cpu().numpy() + K2.cpu().numpy()
            
        x = self.fully_net1(x.reshape(x.shape[0], -1))
        with torch.no_grad():
            x_ = x.view(x.size(0), -1)
            x_ = (x_ > 0).float()
            K = x_ @ x_.t()
            K2 = (1.-x_) @ (1.-x_.t())
            self.K += K.cpu().numpy() + K2.cpu().numpy()
        
        return self.fully_net2(x)


class C8SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(C8SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = escnn.nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.MaskModule(in_type, 29, margin=1),
            escnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = escnn.nn.SequentialModule(
            escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = escnn.nn.SequentialModule(
            escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = escnn.nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net1 = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
        )

        self.fully_net2 = torch.nn.Linear(64, n_classes)
    
    def forward(self, input: torch.Tensor):
        self.K = np.zeros((input.size(0), input.size(0)))

        x = escnn.nn.GeometricTensor(input, self.input_type)
        for block in [self.block1, self.block2, self.pool1, self.block3, self.block4, self.pool2, self.block5, self.block6, self.pool3, self.gpool]:
            x = block(x)
            with torch.no_grad():
                x_ = x.tensor.view(x.tensor.size(0), -1)
                x_ = (x_ > 0).float()
                K = x_ @ x_.t()
                K2 = (1.-x_) @ (1.-x_.t())
                self.K += K.cpu().numpy() + K2.cpu().numpy()

        x = x.tensor
        
        x = self.fully_net1(x.reshape(x.shape[0], -1))
        with torch.no_grad():
            x_ = x.view(x.size(0), -1)
            x_ = (x_ > 0).float()
            K = x_ @ x_.t()
            K2 = (1.-x_) @ (1.-x_.t())
            self.K += K.cpu().numpy() + K2.cpu().numpy()
        
        return self.fully_net2(x)
    


class C8MutantCNN(torch.nn.Module):
    
    def __init__(self, soft=False, n_classes=10):
        
        super(C8MutantCNN, self).__init__()

        self.soft = soft

        self.alphas = torch.autograd.Variable(torch.zeros(10).cuda(), requires_grad=True)
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = escnn.nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.sblock1 = escnn.nn.SequentialModule(
            escnn.nn.MaskModule(in_type, 29, margin=1),
            escnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.sblock1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.sblock2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.spool1 = escnn.nn.SequentialModule(
            escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.sblock2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.sblock3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.sblock3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.sblock4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.spool2 = escnn.nn.SequentialModule(
            escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.sblock4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.sblock5 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.sblock5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = escnn.nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.sblock6 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(out_type),
            escnn.nn.ReLU(out_type, inplace=True)
        )
        self.spool3 = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.sgpool = escnn.nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.sgpool.out_type.size

        widths = [1, 192, 384, 768, 512] #[1, 48, 96, 192, 64]

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[0], widths[1], kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(widths[1]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[1], widths[2], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[2]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[2], widths[2], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[2]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[2], widths[3], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[3]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[3], widths[3], kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(widths[3]),
            torch.nn.ReLU(inplace=True)
        )    
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(widths[3], widths[4], kernel_size=5, padding=1, bias=False),
            torch.nn.BatchNorm2d(widths[4]),
            torch.nn.ReLU(inplace=True)
        )    
        self.pool3 = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=0)

        self.gpool = self.sgpool.export()
        
        # Fully Connected
        self.fully_net1 = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
        )

        self.fully_net2 = torch.nn.Linear(64, n_classes)
    
    def forward(self, input: torch.Tensor):
        self.K = np.zeros((input.size(0), input.size(0)))
        self.KF = np.zeros((input.size(0), input.size(0)))
        self.KU = np.zeros((input.size(0), input.size(0)))
        self.KS = np.zeros((input.size(0), input.size(0)))

        alphas = torch.nn.functional.softmax(torch.stack((self.alphas, torch.zeros_like(self.alphas)), dim=1), dim=1)

        x = escnn.nn.GeometricTensor(input, self.input_type)
        for (i, (ublock, sblock)) in enumerate([(self.block1, self.sblock1),
                                (self.block2, self.sblock2),
                                (self.pool1, self.spool1),
                                (self.block3, self.sblock3),
                                (self.block4, self.sblock4),
                                (self.pool2, self.spool2),
                                (self.block5, self.sblock5),
                                (self.block6, self.sblock6),
                                (self.pool3, self.spool3),
                                (self.gpool, self.sgpool)]):
            xs = sblock(x)
            xu = ublock(x.tensor)
            if self.soft:
                x = alphas[i,0]*xs + alphas[i,1]*escnn.nn.GeometricTensor(xu, sblock.out_type)
            else:
                t = int(alphas[i,0]*xs.tensor.size(dim=1))
                x = escnn.nn.GeometricTensor(torch.cat((
                    torch.narrow(xs.tensor, 1, 0, t), 
                    torch.narrow(xu, 1, 0, xs.tensor.size(dim=1)-t)
                ), dim=1), sblock.out_type)
            with torch.no_grad():
                x_ = x.tensor.view(x.tensor.size(0), -1)
                self.KF += (x_ @ x_.t()).cpu().numpy()
                x_ = (x_ > 0).float()
                K = x_ @ x_.t()
                K2 = (1.-x_) @ (1.-x_.t())
                self.K += K.cpu().numpy() + K2.cpu().numpy()
                xs = xs.tensor.view(xs.tensor.size(0), -1)
                xs = (xs > 0).float()
                K = xs @ xs.t()
                K2 = (1.-xs) @ (1.-xs.t())
                self.KS += K.cpu().numpy() + K2.cpu().numpy()
                xu = xu.view(xu.size(0), -1)
                xu = (xu > 0).float()
                K = xu @ xu.t()
                K2 = (1.-xu) @ (1.-xu.t())
                self.KU += K.cpu().numpy() + K2.cpu().numpy()

        x = x.tensor
        
        x = self.fully_net1(x.reshape(x.shape[0], -1))

        x_ = x.view(x.size(0), -1)
        x_ = (x_ > 0).float()
        K = x_ @ x_.t()
        K2 = (1.-x_) @ (1.-x_.t())
        self.K += K.cpu().numpy() + K2.cpu().numpy()
        
        return self.fully_net2(x)
    

def sgid(index, previous_index = None):
    if previous_index is not None:
        #print(index, previous_index, previous_index[0],  index[0], previous_index[0] == 0, index[0] == 0)
        if previous_index[0] == 0 and index[0] == 0:
            return 2**index[1]
        if previous_index[1] == 0 and index[1] == 0:
            return index[0]
    if index[0] == 0:
        return (None, 2**index[1])
    return (0, 2**index[1])

class EquiCNN(torch.nn.Module):
    
    def __init__(self, reset=False, blocks = [], gs = [(0,0) for _ in range(6)]):
        
        super(EquiCNN, self).__init__()

        self.superspace = gspaces.flipRot2dOnR2(N=8)
        self.gspaces = np.transpose(np.column_stack(([gspaces.trivialOnR2(), gspaces.rot2dOnR2(N=2), gspaces.rot2dOnR2(N=4), gspaces.rot2dOnR2(N=8)], [gspaces.flip2dOnR2(), gspaces.flipRot2dOnR2(N=2), gspaces.flipRot2dOnR2(N=4), gspaces.flipRot2dOnR2(N=8)])))
        print(self.gspaces.shape)
        self.gs = gs
        self.channels = [24, 48, 48, 96, 96, 64]
        self.kernels = [7, 5, 5, 5, 5, 5]
        self.paddings = [1, 2, 2, 2, 2, 1]
        self.blocks = torch.nn.ModuleList(blocks)
        self.architect()
        self.reset = reset #TODO: if true, reinit changed layer, else network morphism 
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.score = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

    def architect(self):
        G, _, _ = self.superspace.restrict(sgid(self.gs[0])) 
        print(self.superspace, sgid(self.gs[0]), G)
        in_type = escnn.nn.FieldType(G, [G.trivial_repr])
        self.input_type = in_type
        for i in range(len(self.gs)):
            out_type = escnn.nn.FieldType(G, int(self.channels[i]/np.sqrt(G.fibergroup.order()/16))*[G.regular_repr])
            print(in_type, out_type)
            if len(self.blocks) <= i:
                self.blocks.append(escnn.nn.SequentialModule(
                    escnn.nn.R2Conv(in_type, out_type, kernel_size=self.kernels[i], padding=self.paddings[i], bias=False),
                    escnn.nn.InnerBatchNorm(out_type),
                    escnn.nn.ReLU(out_type, inplace=True)
                ))
                if i == 1 or i == 3:
                    self.blocks[i].add_module(name="pool", module=escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))                    
            else:
                if in_type != self.blocks[i].in_type or out_type != self.blocks[i].out_type:
                    #print("changing layer", i)
                    #print(self.blocks[i].in_type, "->", in_type)
                    #print(self.blocks[i].out_type, "->", out_type)
                    self.blocks[i] = escnn.nn.SequentialModule(
                        escnn.nn.R2Conv(in_type, out_type, kernel_size=self.kernels[i], padding=self.paddings[i], bias=False),
                        escnn.nn.InnerBatchNorm(out_type),
                        escnn.nn.ReLU(out_type, inplace=True),
                    )
                    if i == 1 or i == 3:
                        self.blocks[i].add_module(name="pool", module=escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))  
            if i < len(self.gs)-1 and self.gs[i] != self.gs[i+1]:
                #print(out_type.gspace, sgid(self.gs[i]), sgid(self.gs[i+1]))
                sg = sgid(self.gs[i+1], self.gs[i])
                restrict = escnn.nn.RestrictionModule(out_type, sg)
                self.blocks[i].add_module(name="restrict", module=restrict)
                G, _, _ = G.restrict(sg)
                #out_type = escnn.nn.FieldType(G, int(self.channels[i]/np.sqrt(G.fibergroup.order()/16))*[G.regular_repr])
                disentangle = escnn.nn.DisentangleModule(restrict.out_type)
                self.blocks[i].add_module(name="disentangle", module=disentangle)
                out_type = disentangle.out_type
            #else: copy old weights
            in_type = out_type
        if len(self.blocks) == len(self.gs):
            self.blocks.append(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0))
            self.blocks.append(escnn.nn.GroupPooling(out_type))
            self.full1 = torch.nn.Sequential(
                torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
            )

            self.full2 = torch.nn.Linear(64, 10)
        else:
            if out_type != self.blocks[len(self.gs)].out_type:
                self.blocks[len(self.gs)] = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
                self.blocks[len(self.gs)+1] = escnn.nn.GroupPooling(out_type)
                self.full1 = torch.nn.Sequential(
                    torch.nn.Linear(self.blocks[-1].out_type.size, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ELU(inplace=True),
                )

    def forward(self, input: torch.Tensor):
        x = escnn.nn.GeometricTensor(input, self.input_type)
        for block in self.blocks:
            x = block(x)
        x = x.tensor
        x = self.full1(x.reshape(x.shape[0], -1))
        return self.full2(x)

    def generate(self):
        candidates = [self.offspring(-1, self.gs[0])]
        for d in range(len(self.gs[0])):
            if self.gs[0][d] < self.gspaces.shape[d]-1:
                g = list(self.gs[0])
                g[d] += 1
                candidates.append(self.offspring(0, tuple(g)))
        for i in range(1, len(self.gs)):
            if self.gs[i][0] < self.gs[i-1][0] or self.gs[i][1] < self.gs[i-1][1]:
                candidates.append(self.offspring(i, self.gs[i-1]))
        return candidates

    def offspring(self, i, G):
        gs = self.gs
        if i >= 0:
            gs[i] = G
            print(self.gs, gs)
        child = EquiCNN(reset = self.reset, blocks = [block for block in self.blocks], gs = gs)
        return child
