#set up unit tests
import unittest
import sys
sys.path.append('../')
import models
import utilities
import torch
import numpy as np


class Test(unittest.TestCase):

    def test_rot(self):
        x = torch.randn(8, 4, 2, 29, 29)
        self.assertTrue(torch.allclose(models.rotate_n(models.rotate_n(x.clone(), 3, 8), 1, 8)[:,:,:,13:16,13:16], models.rotate_n(x.clone(), 1, 2)[:,:,:,13:16,13:16], rtol = 1e-2, atol = 1e-4))

    def test_rotflip(self):
        y = torch.randn(1, 1, 8, 33, 33)**2
        ry = models.rotateflip_n(y.clone(), 1, 4, 1, 2)
        rry = models.rotateflip_n(ry.clone(), 1, 4, 1, 2)
        r1y = models.rotateflip_n(y.clone(), 1, 4, 0, 2)
        r2y = models.rotateflip_n(r1y.clone(), 0, 4, 1, 2)

        self.assertTrue(torch.allclose(ry, r2y))
        self.assertTrue(torch.allclose(y, rry))

    def test_rotflipstack(self):
        y = torch.randn(4, 6, 8, 33, 33)**2
        ry = models.rotateflipstack_n(y.clone(), 1, 4, 1, 2)
        rry = models.rotateflipstack_n(ry.clone(), 1, 4, 1, 2)
        rrry = models.rotateflipstack_n(rry.clone(), 1, 4, 1, 2)
        rrrry = models.rotateflipstack_n(rrry.clone(), 1, 4, 1, 2)

        self.assertTrue(torch.allclose(y, rrrry))

        rfy = models.rotateflipstack_n(y.clone(), 1, 4, 0, 2)
        rfy = models.rotateflipstack_n(rfy.clone(), 0, 4, 1, 2)

        self.assertTrue(torch.allclose(ry, rfy))

        #rotflipstack without flip should work like rotstack
        RFy = models.rotateflipstack_n(y.clone(), 2, 8, 0, 1)
        R_y = models.rotatestack_n(y.clone(), 2, 8)

        self.assertTrue(torch.allclose(R_y, RFy))


    def test_replicate_unit(self):
        for parentgroup in [(0,1), (0,2)]:
            if parentgroup == (0,1):
                childgroups = [(0,0)]
            else:
                childgroups = [(0,0), (0,1)]
            for childgroup in childgroups:
                conversion = int(models.groupsize(parentgroup)/models.groupsize(childgroup))
                parentinchannels = 5
                parentoutchannels = 7
                kernelsize = 3 
                parentlayer = models.GroupConv2d(parentgroup, parentinchannels, parentoutchannels, kernelsize, 0, bias=True)
                reshapein = models.Reshaper(parentinchannels, parentinchannels*conversion, models.groupsize(parentgroup) ,models.groupsize(childgroup), True)
                reshapeout = models.Reshaper(parentoutchannels, parentoutchannels*conversion, models.groupsize(parentgroup), models.groupsize(childgroup), True)
                childlayer = parentlayer.replicate(childgroup, True)
                x = torch.randn(8, parentinchannels, models.groupsize(parentgroup), 9, 9)
                self.assertTrue(torch.allclose(reshapeout(parentlayer(x.clone())), childlayer(reshapein(x.clone())), rtol = 1e-4, atol = 1e-6))
        
    def test_replicate_from_models(self):
        torch.set_printoptions(sci_mode=False)
        for tochange in range(6):
            parent = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)], ordered = True)
            child = parent.offspring(tochange, (0,1))
            print(tochange, parent.gs, child.gs)
            parentlayer = parent.blocks[tochange]._modules["0"]
            childlayer = child.blocks[tochange]._modules["0"]
            if tochange == 5:
                reshapein = child.blocks[tochange-1]._modules["reshaper"]
                x = torch.randn(2, parentlayer.in_channels, models.groupsize(parent.gs[tochange]), 7, 7)
                parentx = parent.blocks[-1](parent.blocks[-2](x.clone()))
                childx = child.blocks[-1](child.blocks[-2](reshapein(x.clone())))
                print(parent.full1(parentx.reshape(parentx.shape[0], -1))[0,:], child.full1(childx.reshape(childx.shape[0], -1))[0,:])
                self.assertTrue(torch.allclose(parent.full1(parentx.reshape(parentx.shape[0], -1)), child.full1(childx.reshape(childx.shape[0], -1)), rtol = 1e-2, atol = 1e-2))
                continue
            reshapeout = parent.blocks[tochange]._modules["reshaper"]
            if tochange == 0:
                x = torch.randn(8, 1, 29, 29)
                self.assertTrue(torch.allclose(reshapeout(parentlayer(x.clone())), childlayer(x.clone()), rtol = 1e-4, atol = 1e-4))
                continue
            reshapein = child.blocks[tochange-1]._modules["reshaper"]
            x = torch.randn(8, parentlayer.in_channels, models.groupsize(parent.gs[tochange]), 9, 9)
            if not torch.allclose(reshapeout(parentlayer(x.clone())), childlayer(reshapein(x.clone())), rtol = 1e-4, atol = 1e-4):
                print(reshapeout(parentlayer(x.clone()))[0,0:6,:,0,0])
                print(childlayer(reshapein(x.clone()))[0,0:6,:,0,0])
            self.assertTrue(torch.allclose(reshapeout(parentlayer(x.clone())), childlayer(reshapein(x.clone())), rtol = 1e-4, atol = 1e-4))

    def test_replicate_complete(self):
        for tochange in range(6):
            print(tochange)
            parent = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)], ordered = True)
            child = parent.offspring(tochange, (0,1))
            x = torch.randn(8, 1, 29, 29)
            self.assertTrue(torch.allclose(parent(x.clone()), child(x.clone()), rtol = 1e-1, atol = 1e-1))
        
    def test_replicate_independent_objects(self):
        for tochange in range(6):
            print(tochange)
            parent = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)], ordered = True)
            child = parent.offspring(tochange, (0,1))
            x = torch.randn(8, 1, 29, 29)
            before = parent(x.clone())
            child.train()
            y = child(x.clone())
            loss = child.loss_function(y, torch.rand_like(y))
            loss.backward()
            child.optimizer.step()
            after = parent(x.clone())
            self.assertTrue(not torch.allclose(before, after, rtol = 1e-4, atol = 1e-6))

    def test_lifting_equivariance(self):
        in_channels = 1
        out_channels = 1
        kernel_size = 3
        batchsize = 1
        S = 3
        for group in [(1,0), (0,0), (1,2), (0,2)]:
            print(group)

            layer = models.LiftingConv2d(group, in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
            layer.eval()
            for i in range(kernel_size):
                for j in range(kernel_size):
                    layer.weight.data[0,0,i,j] = (i-1)*10+j-1

            x = torch.randn(batchsize, in_channels, S, S)
            for i in range(S):
                for j in range(S):
                    x[0,0,i,j] = (i-1)*10+j-1
            # the input image belongs to the space X, so we use the original action to rotate it
            gx = models.rotateflip_n(x, 1, 2**group[1], 1, 2**group[0])

            
            # compute the output
            psi_x = layer(x)
            psi_gx = layer(gx)

            # the output is a function in the space Y, so we need to use the new action to rotate it
            g_psi_x = models.rotateflipstack_n(psi_x, 1, 2**group[1], 1, 2**group[0])

            self.assertTrue(psi_x.shape == g_psi_x.shape)
            self.assertTrue(psi_x.shape == (batchsize, out_channels, models.groupsize(group), S, S))

            # check the model is giving meaningful outputs
            self.assertTrue(not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4))

            # check equivariance
            if group == (1,2):
                print(x)
                print(gx)
                print(psi_x)
                print(psi_gx)
                print(g_psi_x)
            print(torch.allclose(psi_gx, g_psi_x, atol=1e-6, rtol=1e-6))

            # check the model has the right number of parameters
            self.assertTrue(layer.weight.numel() == in_channels * out_channels * kernel_size**2)
            self.assertTrue(layer.bias.numel() == out_channels)

    def test_groupconv_equivariance(self):
        torch.manual_seed(0)
        in_channels = 8
        out_channels = 8
        kernel_size = 3
        batchsize = 6
        S = 3
        for group in [(1,0), (0,0), (1,1), (0,1), (1,2), (0,2)]:

            layer = models.GroupConv2d(group, in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
            layer.eval()

            x = torch.randn(batchsize, in_channels, models.groupsize(group), S, S)

            psi_x = layer(x)

            for f in range(2**group[0]):
                for r in range(2**group[1]):
                    gx = models.rotateflipstack_n(x, r, 2**group[1], f, 2**group[0])

                    # compute the output
                    psi_gx = layer(gx)

                    # the output is a function in the space Y, so we need to use the new action to rotate it
                    g_psi_x = models.rotateflipstack_n(psi_x, r, 2**group[1], f, 2**group[0])

                    self.assertTrue(psi_x.shape == g_psi_x.shape)
                    self.assertTrue(psi_x.shape == (batchsize, out_channels, models.groupsize(group), S, S))

                    # check the model is giving meaningful outputs
                    self.assertTrue(not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4))

                    # if group == (1,2):
                    #     print(x)
                    #     print(gx)
                    #     print(psi_x)
                    #     print(psi_gx)
                    #     print(g_psi_x)

                    #score += sum([torch.allclose(psi_gx[:,:,i,:,:],g_psi_x[:,:,i,:,:]) for i in range(psi_gx.shape[2])])

                    # check equivariance
                    self.assertTrue(torch.allclose(psi_gx, g_psi_x, atol=1e-5, rtol=1e-5))
                    if not torch.allclose(psi_gx, g_psi_x, atol=1e-5, rtol=1e-5):
                        print(group, (f,r), [torch.allclose(psi_gx[:,:,i,:,:],g_psi_x[:,:,i,:,:], atol=1e-4, rtol=1e-4) for i in range(psi_gx.shape[2])])
                        #eq = False
                    else:
                        print(group, (f,r), "equivariant")

            self.assertTrue(layer.weight.numel() == in_channels * out_channels * models.groupsize(group) * kernel_size**2)
            self.assertTrue(layer.bias.numel() == out_channels)

        
        #print(bestperm, bestscore)

    def test_mixedgroupconv(self):
        torch.manual_seed(0)
        torch.set_printoptions(precision=2, sci_mode=False)
        in_channels = 2
        out_channels = 2
        kernel_size = 3
        batchsize = 4
        S = 3
        for group in [(1,0), (0,0), (1,1), (0,1), (1,2), (0,2), (1,3), (0,3), (1,4), (0,4)]:
        #for group in [(1,2)]:
            layer = models.MixedGroupConv2dV2(group, in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=int(kernel_size//2), bias=True)#, test=True)
            layer.eval()
            
            x = torch.rand(batchsize, in_channels*models.groupsize(group), S, S)
            for flip in range(group[0]+1):
                for rotation in range(group[1]+1):
                    for k in range(len(layer.alphas)):
                        if layer.groups[k][0] >= flip and layer.groups[k][1] >= rotation:
                            layer.alphas.data[k] = 0
                        else:
                            layer.alphas.data[k] = -np.inf
                    gx = models.rotateflipstack_n(x.clone().reshape(batchsize, in_channels, models.groupsize(group), S, S), 2**(group[1]-rotation)%2**group[1], 2**group[1], flip, 2**group[0]).reshape(batchsize, in_channels*models.groupsize(group), S, S)

                    psi_x = layer(x.clone())
                    psi_gx = layer(gx.clone())

                    g_psi_x = models.rotateflipstack_n(psi_x.clone().reshape(batchsize, out_channels, models.groupsize(group), S, S), 2**(group[1]-rotation)%2**group[1], 2**group[1], flip, 2**group[0]).reshape(batchsize, out_channels*models.groupsize(group), S, S)

                    self.assertTrue(psi_x.shape == g_psi_x.shape)
                    self.assertTrue(psi_x.shape == (batchsize, out_channels*models.groupsize(group), S, S))

                    self.assertTrue(not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4))

                    if not torch.allclose(psi_gx, g_psi_x, atol=1e-4, rtol=1e-4):
                        print(group, (flip, rotation), (flip,2**(group[1]-rotation)%2**group[1]), [torch.allclose(psi_gx[:,i,:,:],g_psi_x[:,i,:,:], atol=1e-4, rtol=1e-4) for i in range(psi_gx.shape[1])])
                        #eq = False
                    else:
                        print(group, (flip, rotation), (flip,2**(group[1]-rotation)%2**group[1]), "equivariant")
                    #self.assertTrue(torch.allclose(psi_gx, g_psi_x, atol=1e-4, rtol=1e-6))
            
    def test_DEANASNet(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.randn(16, 1, 29, 29)
        labels = torch.randint(0, 9, (16,))
        model = models.DEANASNet(superspace=(1,2), stages = 6)
        model(inputs)
        model.train()
        model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs0 = model(inputs)
        loss = model.loss_function(outputs0, labels)
        alphas0 = model.blocks[1]._modules["0"].alphas.clone()
        norms0 = model.blocks[1]._modules["0"].norms.clone()
        weights0 =  model.blocks[0]._modules["0"].weights[0].clone()
        model.optimizer.zero_grad()
        model.alphaopt.zero_grad()
        loss.backward()
        model.optimizer.step()
        self.assertTrue(torch.allclose(alphas0, model.blocks[1]._modules["0"].alphas, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(norms0, model.blocks[1]._modules["0"].norms, atol=1e-7, rtol=1e-7))
        weights1 =  model.blocks[0]._modules["0"].weights[0].clone()
        self.assertTrue(not torch.allclose(weights0, weights1, atol=1e-7, rtol=1e-7))
        model.alphaopt.step()
        self.assertTrue(not torch.allclose(alphas0, model.blocks[1]._modules["0"].alphas, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(weights1, model.blocks[0]._modules["0"].weights[0], atol=1e-7, rtol=1e-7))

    def test_offspring_DEANAS(self):
        torch.manual_seed(0)
        torch.set_printoptions(sci_mode=False)
        model = models.DEANASNet(superspace=(1,2), stages = 2, basechannels=1, discrete=True)
        child = model.offspring(len(model.channels)-1, (1,1), verbose=True)
        xmodel = torch.randn(16, 1, 29, 29)
        xchild = xmodel.clone()
        for i in range(len(model.blocks)):
            xmodel = model.blocks[i](xmodel)
            xchild = child.blocks[i](xchild)
            if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
                print(i)
                #if i == 7:
                #    print(xmodel[0:4,:,0:6,0])
                #    print(xchild[0:4,:,0:6,0])
            #self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4))
            if i == len(model.blocks)-3:
                xmodel = xmodel.reshape(xmodel.shape[0], -1)
                xchild = xchild.reshape(xchild.shape[0], -1)
    
    # def test_offspring_DEANAS_all(self):
    #     torch.manual_seed(0)
    #     torch.set_printoptions(sci_mode=False)
    #     for f in range(2):
    #         for r in range(3):
    #             for sf in range(f+1):
    #                 for sr in range(r+1):
    #                     if sf != f or sr != r:
    #                         model = models.DEANASNet(superspace=(f,r), stages = 2, basechannels=2, discrete=True)
    #                         child = model.offspring(len(model.channels)-1, (sf,sr))
    #                         xmodel = torch.randn(4, 1, 29, 29)
    #                         xchild = xmodel.clone()
    #                         for i in range(8):
    #                             xmodel = model.blocks[i](xmodel)
    #                             xchild = child.blocks[i](xchild)
    #                         if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
    #                             print((f,r), (sf,sr))
    #                         else:
    #                             print((f,r), (sf,sr), "passed")

    def test_offspring_DEANAS_all(self):
        torch.manual_seed(0)
        torch.set_printoptions(sci_mode=False)
        mf = 1
        mr = 2
        model = models.DEANASNet(superspace=(mf,mr), stages = 2, basechannels=2, discrete=True)
        while max(mf, mr) > 0:
            for f in range(mf+1):
                for r in range(mr+1):
                    if mf == f and mr == r:
                        continue
                    child = model.offspring(len(model.channels)-1, (f,r))
                    xmodel = torch.randn(4, 1, 29, 29)
                    xchild = xmodel.clone()
                    for i in range(8):
                        xmodel = model.blocks[i](xmodel)
                        xchild = child.blocks[i](xchild)
                    if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
                        print((mf,mr), (f,r))
                    else:
                        print((mf,mr), (f,r),"passed")
            if mf == 1:
                mf = 0
            elif mr > 0:
                mr -= 1
            model = model.offspring(len(model.channels)-1, (mf,mr))
        mf = 1
        mr = 2
        model = models.DEANASNet(superspace=(mf,mr), stages = 2, basechannels=2, discrete=True)
        while max(mf, mr) > 0:
            for f in range(mf+1):
                for r in range(mr+1):
                    if mf == f and mr == r:
                        continue
                    child = model.offspring(len(model.channels)-1, (f,r))
                    xmodel = torch.randn(4, 1, 29, 29)
                    xchild = xmodel.clone()
                    for i in range(8):
                        xmodel = model.blocks[i](xmodel)
                        xchild = child.blocks[i](xchild)
                    if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
                        print((mf,mr), (f,r))
                    else:
                        print((mf,mr), (f,r),"passed")
            if mr > 1:
                mr -= 1
            elif mf == 1:
                mf = 0
            elif mr == 1:
                mr = 0
            model = model.offspring(len(model.channels)-1, (mf,mr))
        mf = 1
        mr = 2
        model = models.DEANASNet(superspace=(mf,mr), stages = 2, basechannels=2, discrete=True)
        while max(mf, mr) > 0:
            for f in range(mf+1):
                for r in range(mr+1):
                    if mf == f and mr == r:
                        continue
                    child = model.offspring(len(model.channels)-1, (f,r))
                    xmodel = torch.randn(4, 1, 29, 29)
                    xchild = xmodel.clone()
                    for i in range(8):
                        xmodel = model.blocks[i](xmodel)
                        xchild = child.blocks[i](xchild)
                    if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
                        print((mf,mr), (f,r))
                    else:
                        print((mf,mr), (f,r),"passed")
            if mr > 0:
                mr -= 1
            elif mf == 1:
                mf = 0
            model = model.offspring(len(model.channels)-1, (mf,mr))

    def test_offspring_DEANAS_lifting(self):
        torch.manual_seed(0)
        torch.set_printoptions(sci_mode=False)
        model = models.DEANASNet(superspace=(0,2), stages = 2, basechannels=1, discrete=True)
        for i in range(len(model.channels)-1):
            model = model.offspring(len(model.channels)-1-i, (0,1))
        child = model.offspring(0, (0,1))
        xmodel = torch.randn(16, 1, 29, 29)
        xchild = xmodel.clone()
        for i in range(len(model.blocks)):
            xmodel = model.blocks[i](xmodel)
            xchild = child.blocks[i](xchild)
            if not torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4):
                print(i)
                if i == 0:
                    print(xmodel[0:4,:,0:6,0])
                    print(xchild[0:4,:,0:6,0])
            #self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4))
            if i == len(model.blocks)-3:
                xmodel = xmodel.reshape(xmodel.shape[0], -1)
                xchild = xchild.reshape(xchild.shape[0], -1)


    def test_offspring(self):
        torch.set_printoptions(sci_mode=False)
        tochange = 4
        model = models.TDRegEquiCNN(gs = [(0,2) for _ in range(tochange+1)]+[(0,1) for _ in range(5-tochange)], ordered = True)
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
        xmodel = torch.randn(16, 1, 29, 29)
        xchild = xmodel.clone()
        for i in range(len(model.blocks)):
            #print("modules", i, model.blocks[i]._modules.keys(), child.blocks[i]._modules.keys())
            if "0" in model.blocks[i]._modules.keys() and i > 0 and i != tochange: #model.blocks[i]._modules["0"].weight.shape != child.blocks[i]._modules["0"].weight.shape:
                child_filter, child_x, child_bias = child.blocks[i]._modules["0"].test_filter_x(xchild)
                model_filter, model_x, model_bias = model.blocks[i]._modules["0"].test_filter_x(xmodel)
                print("model_filter:", model_filter[0:4,0:8,0,0])
                print("child_filter:", child_filter[0:4,0:8,0,0])
                print("model_x:", model_x.shape, model_x[0,0,0,:])
                print("child_x:", child_x.shape, child_x[0,0,0,:])
                print("xmodel:", xmodel.shape, xmodel[1,1,1,:])
                print("xchild:", xchild.shape, xchild[1,1,1,:])
                print(i)
                self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_filter, child_filter, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_x, child_x, rtol = 1e-4, atol = 1e-4))
                self.assertTrue(torch.allclose(model_bias, child_bias, rtol = 1e-4, atol = 1e-4))
                #self.assertTrue(torch.allclose(model.blocks[i](xmodel), child.blocks[i](xchild), rtol = 1e-4, atol = 1e-4))
            xmodel = model.blocks[i](xmodel)
            xchild = child.blocks[i](xchild)
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
                if "0" in model.blocks[i]._modules.keys():
                    print("weight shapes", model.blocks[i]._modules["0"].weight.shape, child.blocks[i]._modules["0"].weight.shape)
        print(xmodel.reshape(xmodel.shape[0], -1)[0:4,0:4])
        print(xchild.reshape(xchild.shape[0], -1)[0:4,0:4])
        print(model.full1._modules["0"](xmodel.reshape(xmodel.shape[0], -1))[0:4,0:4])
        print(child.full1._modules["0"](xchild.reshape(xchild.shape[0], -1))[0:4,0:4])
        xmodel = model.full1(xmodel.reshape(xmodel.shape[0], -1))
        xchild = child.full1(xchild.reshape(xchild.shape[0], -1))
        print(xmodel[1,:])
        print(xchild[1,:])
        self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-6))
        xmodel = model.full2(xmodel)
        xchild = child.full2(xchild)
        self.assertTrue(torch.allclose(xmodel, xchild, rtol = 1e-4, atol = 1e-6))

    def test_equivariance(self):
        in_channels = 5
        out_channels = 9
        kernel_size = 3
        batchsize = 4
        S = 17
        for g in [1,2]:
            layer = models.GroupConv2d(group=(0,g), in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
            layer.eval()

            x = torch.randn(batchsize, in_channels, 2**g, S, S)**2
            gx = models.rotatestack_n(x, 1, 2**g)

            psi_x = layer(x)
            psi_gx = layer(gx)

            g_psi_x = models.rotatestack_n(psi_x, 1, 2**g)

            assert psi_x.shape == g_psi_x.shape
            assert psi_x.shape == (batchsize, out_channels, 2**g, S, S)

            assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

            assert torch.allclose(psi_gx, g_psi_x, atol=1e-5, rtol=1e-5)

            assert layer.weight.numel() == in_channels * out_channels * 2**g * kernel_size**2
            assert layer.bias.numel() == out_channels


            layer = models.LiftingConv2d((0,g), in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
            layer.eval()

            x = torch.randn(batchsize, in_channels, S, S)
            gx = models.rotate_n(x, 1, 2**g)

            psi_x = layer(x)
            psi_gx = layer(gx)

            g_psi_x = models.rotatestack_n(psi_x, 1, 2**g)

            assert psi_x.shape == g_psi_x.shape
            assert psi_x.shape == (batchsize, out_channels, 2**g, S, S)

            assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

            assert torch.allclose(psi_gx, g_psi_x, atol=1e-6, rtol=1e-6)

            assert layer.weight.numel() == in_channels * out_channels * kernel_size**2
            assert layer.bias.numel() == out_channels


    def test_train(self):
        self.model.train()
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.model.loss_function(outputs, labels)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

    def test_eval(self):
        self.model.eval()
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.model.loss_function(outputs, labels)
            self.assertTrue(loss.item() <= 0)

    def test_generate(self):
        allgs = [[(0,2) for _ in range(i)]+[(0,1) for _ in range(6-i)] for i in range(6)] + \
                [[(0,2) for _ in range(i)]+[(0,0) for _ in range(6-i)] for i in range(6)]

        for gs in allgs:
            model = models.TDRegEquiCNN(gs=gs)
            children = model.generate()
    


    
if __name__ == '__main__':
    unittest.main()
