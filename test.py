#set up unit tests
import unittest
import sys
sys.path.append('../')
import models
import utilities
import torch
import numpy as np

class TestTDRegEquiCNN(unittest.TestCase):
    def setUp(self):
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.TDRegEquiCNN()
        #self.model = self.model.to(self.device)
        #self.model.optimizer = torch.optim.SGD(self.model.parameters(), lr=5e-4)
        #self.train_loader, _, _ = utilities.get_dataloaders(path_to_dir="../")
    
    # def test_adapt(self):
    #     tochange = 5
    #     model = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)])
    #     # parent.blocks = torch.nn.ModuleList([models.LiftingConv2d((0,2), 1, 3, 3, 0, bias=True),
    #     #                                     models.GroupConv2d((0,2), 3, 4, 3, 0, bias=True),
    #     #                                     models.Reshaper(4,8,4,2),
    #     #                                     models.GroupConv2d((0,1), 8, 6, 3, 0, bias=True)])
    #     # parent.full1 = torch.nn.Linear(108, 64)
    #     # parent.full2 = torch.nn.Linear(64, 10)
    #     child = model.offspring(tochange, (0,1))

    #     xmodel = torch.randn(2, 1, 29, 29)
    #     xchild = xmodel.clone()
        


    def test_offspring(self):
        torch.set_printoptions(sci_mode=False)
        tochange = 4
        model = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)], ordered = False)
        # for i in range(len(model.blocks)):
        #     if "0" in model.blocks[i]._modules.keys():
        #         # model.blocks[i]._modules["0"].weight = torch.nn.Parameter(torch.zeros_like(model.blocks[i]._modules["0"].weight))
        #         # print(i, model.blocks[i]._modules["0"].weight.shape)
        #         # if i > 0:
        #         #     for j in range(model.blocks[i]._modules["0"].weight.shape[2]):
        #         #         model.blocks[i]._modules["0"].weight.data[:,:,j,2,2] += 1
        #         # else:
        #         #     for j in range(model.blocks[i]._modules["0"].weight.shape[1]):
        #         #         model.blocks[i]._modules["0"].weight.data[:,j,3,3] += 1 
        #         if i == tochange:
        #             model.blocks[i]._modules["0"].weight = torch.nn.Parameter(torch.zeros_like(model.blocks[i]._modules["0"].weight))
        #             for j in range(model.blocks[i]._modules["0"].weight.shape[2]):
        #                 for k in range(model.blocks[i]._modules["0"].weight.shape[0]):
        #                     for l in range(model.blocks[i]._modules["0"].weight.shape[1]):
        #                         model.blocks[i]._modules["0"].weight.data[k,l,j,:,:] += j + k*100 + l*10
        child = model.offspring(tochange, (0,1))
        #print(model.gs)
        #print(child.gs)
        xmodel = torch.ones(2, 1, 29, 29)
        xchild = xmodel.clone()
        for i in range(len(model.blocks)):
            #print("modules", i, model.blocks[i]._modules.keys(), child.blocks[i]._modules.keys())
            if "0" in model.blocks[i]._modules.keys() and i == 5: #model.blocks[i]._modules["0"].weight.shape != child.blocks[i]._modules["0"].weight.shape:
                child_filter, child_x, child_bias = child.blocks[i]._modules["0"].test_filter_x(xchild)
                model_filter, model_x, model_bias = model.blocks[i]._modules["0"].test_filter_x(xmodel)
                print("model:", model.blocks[i]._modules["0"].weight.shape, model_filter.shape, model.blocks[i]._modules["0"].weight[:,0,:,0,0], model_filter[:,0,0,0], model_filter[0:4,0:8,0,0])
                print("child:", child.blocks[i]._modules["0"].weight.shape, child_filter.shape, child.blocks[i]._modules["0"].weight[:,0,:,0,0], child_filter[:,0,0,0], child_filter[0:4,0:8,0,0])
                print("model_x:", model_x.shape, model_x[0,0,0,:])
                print("child_x:", child_x.shape, child_x[0,0,0,:])
                print("model_bias:", model_bias.shape, model_bias)
                print("child_bias:", child_bias.shape, child_bias)
                self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_filter, child_filter, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_x, child_x, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_bias, child_bias, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model.blocks[i](xmodel), child.blocks[i](xchild), rtol = 1e-4, atol = 1e-4))
            xmodel = model.blocks[i](xmodel)
            xchild = child.blocks[i](xchild)
            #print("x shapes", xmodel.shape, xchild.shape)
            if xmodel.shape != xchild.shape:
                out_channels = xchild.shape[1]
                in_groupsize = xmodel.shape[2]
                out_groupsize = xchild.shape[2]
                in_order = [int('{:0{width}b}'.format(n, width=int(np.log2(in_groupsize)))[::-1], 2) for n in range(in_groupsize)]
                out_order = [int('{:0{width}b}'.format(n, width=int(np.log2(out_groupsize)))[::-1], 2) for n in range(out_groupsize)]
                xmodel_rs = xmodel[:,:,in_order].view(-1, out_channels, out_groupsize, xmodel.shape[-2], xmodel.shape[-1])[:,:,out_order]
            else:
                xmodel_rs = xmodel
            if not torch.allclose(xmodel_rs, xchild):
                #mockxmodel = torch.zeros_like(xmodel)
                #mockxchild = torch.zeros_like(xchild)
                if "0" in model.blocks[i]._modules.keys():
                    print("weight shapes", model.blocks[i]._modules["0"].weight.shape, child.blocks[i]._modules["0"].weight.shape)
                #print("x sample", xmodel_rs[0,0,0], xchild[0,0,0])
                #print("diff:", (xmodel_rs-xchild).abs().max())
            if i == -1:
                xmodel = xmodel * 0 + 1
                xchild = xchild * 0 + 1
            elif i >= 4:
                print(i, (xmodel-xchild).abs().sum())
                print(i, "xmodel:", xmodel.shape)
                print(i, "xchild:", xchild.shape)
        print(xmodel.reshape(xmodel.shape[0], -1)[0,:])
        print(xchild.reshape(xchild.shape[0], -1)[0,:])
        xmodel = model.full1(xmodel.reshape(xmodel.shape[0], -1))
        xchild = child.full1(xchild.reshape(xchild.shape[0], -1))
        self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4))
        xmodel = model.full2(xmodel)
        xchild = child.full2(xchild)
        self.assertTrue(torch.allclose(xmodel, xchild))

    # def test_equivariance(self):
    #     in_channels = 5
    #     out_channels = 9
    #     kernel_size = 3
    #     batchsize = 4
    #     S = 17
    #     for g in [1,2]:
    #         layer = models.GroupConv2d(group=(0,g), in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
    #         layer.eval()

    #         x = torch.randn(batchsize, in_channels, 2**g, S, S)**2
    #         gx = models.rotatestack_n(x, 1, 2**g)

    #         psi_x = layer(x)
    #         psi_gx = layer(gx)

    #         g_psi_x = models.rotatestack_n(psi_x, 1, 2**g)

    #         assert psi_x.shape == g_psi_x.shape
    #         assert psi_x.shape == (batchsize, out_channels, 2**g, S, S)

    #         assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

    #         assert torch.allclose(psi_gx, g_psi_x, atol=1e-5, rtol=1e-5)

    #         assert layer.weight.numel() == in_channels * out_channels * 2**g * kernel_size**2
    #         assert layer.bias.numel() == out_channels


    #         layer = models.LiftingConv2d((0,g), in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
    #         layer.eval()

    #         x = torch.randn(batchsize, in_channels, S, S)
    #         gx = models.rotate_n(x, 1, 2**g)

    #         psi_x = layer(x)
    #         psi_gx = layer(gx)

    #         g_psi_x = models.rotatestack_n(psi_x, 1, 2**g)

    #         assert psi_x.shape == g_psi_x.shape
    #         assert psi_x.shape == (batchsize, out_channels, 2**g, S, S)

    #         assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

    #         assert torch.allclose(psi_gx, g_psi_x, atol=1e-6, rtol=1e-6)

    #         assert layer.weight.numel() == in_channels * out_channels * kernel_size**2
    #         assert layer.bias.numel() == out_channels


    # def test_train(self):
    #     self.model.train()
    #     for inputs, labels in self.train_loader:
    #         inputs = inputs.to(self.device)
    #         labels = labels.to(self.device)
    #         outputs = self.model(inputs)
    #         loss = self.model.loss_function(outputs, labels)
    #         self.model.optimizer.zero_grad()
    #         loss.backward()
    #         self.model.optimizer.step()

    # def test_eval(self):
    #     self.model.eval()
    #     for inputs, labels in self.train_loader:
    #         inputs = inputs.to(self.device)
    #         labels = labels.to(self.device)
    #         outputs = self.model(inputs)
    #         loss = self.model.loss_function(outputs, labels)
    #         self.assertTrue(loss.item() <= 0)

    # def test_generate(self):
    #     allgs = [[(0,2) for _ in range(i)]+[(0,1) for _ in range(6-i)] for i in range(6)] + \
    #             [[(0,2) for _ in range(i)]+[(0,0) for _ in range(6-i)] for i in range(6)]

    #     for gs in allgs:
    #         model = models.TDRegEquiCNN(gs=gs)
    #         children = model.generate()
    


    
if __name__ == '__main__':
    unittest.main()
