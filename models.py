import torch
import escnn
from escnn import gspaces
import numpy as np


class UnsteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, width_equated = True):
        
        super(UnsteerableCNN, self).__init__()    

        if width_equated:
            widths = [1, 192, 384, 786, 64]
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
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        
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
        self.pool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2)

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
        self.pool3 = torch.nn.AvgPool2d(kernel_size=3, stride=1)

        self.gpool = torch.nn.AdaptiveAvgPool2d((1,1))
        
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(widths[4], widths[4]),
            torch.nn.BatchNorm1d(widths[4]),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(widths[4], n_classes),
        )

    def forward(self, x: torch.Tensor):
        self.K = np.zeros((x.size(0), x.size(0)))

        for block in [self.block1, self.block2, self.pool1, self.block3, self.block4, self.pool2, self.block5, self.block6, self.pool3, self.gpool]:
            x = block(x)
            x_ = x.tensor.view(x.tensor.size(0), -1)
            x_ = (x_ > 0).float()
            K = x_ @ x_.t()
            K2 = (1.-x_) @ (1.-x_.t())
            self.K += K.cpu().numpy() + K2.cpu().numpy()

        x = x.tensor
        
        x = self.fully_net1(x.reshape(x.shape[0], -1))

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
            x_ = x.tensor.view(x.tensor.size(0), -1)
            x_ = (x_ > 0).float()
            K = x_ @ x_.t()
            K2 = (1.-x_) @ (1.-x_.t())
            self.K += K.cpu().numpy() + K2.cpu().numpy()

        x = x.tensor
        
        x = self.fully_net1(x.reshape(x.shape[0], -1))

        x_ = x.view(x.size(0), -1)
        x_ = (x_ > 0).float()
        K = x_ @ x_.t()
        K2 = (1.-x_) @ (1.-x_.t())
        self.K += K.cpu().numpy() + K2.cpu().numpy()
        
        return self.fully_net2(x)
    