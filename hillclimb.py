import datetime
import pickle
import torch
import numpy as np
import models
import utilities
from operator import attrgetter
import argparse
import copy
import pathlib

class HillClimber(object):
    def __init__(self, reset = True, allkids = False, reg = False, skip = False, baselines = False, lr = 0.1, datapath = ".."):
        if baselines:
            exp = "bs"
        else:
            exp = "hc"
        self.filename = './out/logs'+exp+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
        print(self.filename)
        self.ordered = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.validation_loader, self.test_loader = utilities.get_dataloaders(path_to_dir=datapath)
        self.reg = reg
        if reg:
            if skip:
                self.model = models.SkipEquiCNN(gs=[(0,4) for _ in range(6)], ordered = self.ordered, lr = lr)
            else:
                self.model = models.TDRegEquiCNN(gs=[(0,2) for _ in range(6)], ordered = self.ordered, lr = lr)
        else:
            self.model = models.EquiCNN(reset)
        self.skip = True
        self.options = []
        self.allkids = allkids
        self.history = {}

    def train(self, epochs = 1, start = 0, lr = 5e-4):
        if len(self.options)==0:
            totrain = [self.model]
        else:
            totrain = self.options
        for model in totrain:
            model = model.to(self.device)
            #model.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            dataloaders = {
                "train": self.train_loader,
                "validation": self.validation_loader
            }
            if model.uuid not in self.history:
                if model.parent is None or model.parent not in self.history:
                    self.history[model.uuid] = {'train': {'loss': [], 
                                        'accuracy': [], 
                                        'batch': [], 
                                        'batchloss': []},
                              'trainsteps': [],
                              'validation' : {'loss': [], 
                                              'accuracy': []},
                              'epochsteps': [],
                              'ghistory': []}
                else:
                    self.history[model.uuid] = copy.deepcopy(self.history[model.parent])
            self.history[model.uuid]["epochsteps"] += np.linspace(start, start+1, epochs, endpoint=False).tolist()
            self.history[model.uuid]["ghistory"].append(model.gs)
            for _ in range(epochs):
                for phase in ['train', 'validation']:
                    batch = []
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    running_corrects = 0
                    running_count = 0

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(inputs)
                        loss = model.loss_function(outputs, labels)

                        if phase == 'train':
                            model.optimizer.zero_grad()
                            loss.backward()
                            model.optimizer.step()

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.detach() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        running_count += inputs.size(0)
                        if phase == "train":
                            batch.append(running_count)
                            self.history[model.uuid][phase]['batchloss'].append(loss.detach().item())
                    epoch_loss = running_loss / running_count
                    epoch_acc = running_corrects.float() / running_count

                    self.history[model.uuid][phase]['loss'].append(epoch_loss.item())
                    self.history[model.uuid][phase]['accuracy'].append(epoch_acc.item())
                    model.score = epoch_acc.item()

                    if phase == "train":
                        self.history[model.uuid]["trainsteps"] += [b / (running_count+epochs) + start for b in batch]
                    #else:
                    #    print(epoch_acc.item())
            model = model.to("cpu")

    def validate(self, model):
        model = model.to(self.device)
        dataloaders = {
            "validation": self.validation_loader
        }
        phase = "validation"
        model.eval()

        running_corrects = 0
        running_count = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_count += inputs.size(0)
            
        acc = running_corrects.float() / running_count

        return acc.item()

    def test(self, model1, model2):
        model1 = model1.to(self.device)
        model2 = model2.to(self.device)
        dataloaders = {
            "validation": self.validation_loader
        }
        phase = "validation"
        model1.eval()
        model2.eval()

        if model1.gs == model2.gs:
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)

                assert torch.allclose(outputs1, outputs2, atol=1e-5, rtol=1e-5)
    
    def saveargs(self, args):
        self.history["args"] = args
    
    def generate(self):
        if self.allkids:
            children = []
            #children += self.model.generate()
            print("parent:", self.model.gs, self.validate(self.model))
            for child in self.model.generate():
                print("child:", child.gs, self.validate(child))
                self.test(self.model, child)
                children.append(child)
            for model in self.options:
                #children += model.generate()
                print("parent:", model.gs, self.validate(model))
                for child in model.generate():
                    print("child:", child.gs, self.validate(child))
                    self.test(model, child)
                    children.append(child)
            self.options = children
        else:
            self.options = self.model.generate()

    def select(self):
        for child in sorted(self.options, key=attrgetter('score'), reverse=True):
            print(child.gs, sum(p.numel() for p in child.parameters() if p.requires_grad), child.score)
        self.model = max(self.options, key=attrgetter("score"))
        if self.allkids:
            bests = {}
            for child in self.options:
                if str(child.gs) not in bests or child.score > bests[str(child.gs)]["score"]:
                    bests[str(child.gs)] = {"uuid": child.uuid, "score": child.score}
            uuids = [bests[g]["uuid"] for g in bests.keys()]
            print(len(self.options), len(uuids), end = " ")
            for child in self.options:
                if child.uuid not in uuids:
                    self.options.remove(child)
            print(len(self.options))

    def save(self):
        #for model in self.options:
            #self.history[model.uuid] = self.history[model.uuid]
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)

    def hillclimb(self, iterations = -1, epochs = 5, lr = 5e-4):
        self.train(epochs = epochs, start = 0)
        for iter in range(iterations):
            self.generate()
            print("Iteration ", iter)
            self.train(epochs = epochs, start = iter+1, lr = lr)
            self.select()
            self.save()

    def baselines(self, iterations = -1, epochs = 40, lr = 5e-4):
        upper = 5
        allgs = [[(0,i) for _ in range(6)] for i in range(upper)]
        for j in range(upper-1):
            allgs += [[(0,i) for _ in range(5)]+[(0,j)] for i in range(1,upper) if i >= j]
            for k in range(j+1):
                allgs += [[(0,i) for _ in range(4)]+[(0,j)]+[(0,k)] for i in range(1,upper) if i >= j]
                for l in range(k+1):
                    allgs += [[(0,i) for _ in range(3)]+[(0,j)]+[(0,k)]+[(0,l)] for i in range(1,upper) if i >= j]
                    for m in range(l+1):
                        allgs += [[(0,i) for _ in range(2)]+[(0,j)]+[(0,k)]+[(0,l)]+[(0,m)] for i in range(1,upper) if i >= j]
                        for n in range(m+1):
                            allgs += [[(0,i) for _ in range(1)]+[(0,j)]+[(0,k)]+[(0,l)]+[(0,m)]+[(0,n)] for i in range(1,upper) if i >= j]
        lastgs = []
        for gs in sorted(allgs):
            if gs == lastgs:
                continue
            lastgs = gs
            if self.reg:
                if self.skip:
                    model = models.SkipEquiCNN(gs=gs, ordered = self.ordered, lr = lr)
                else:
                    model = models.TDRegEquiCNN(gs=gs, ordered = self.ordered, lr = lr)
            else:
                model = models.EquiCNN(reset=False, gs = gs)
            self.options.append(model)
        for iter in range(iterations):
            self.train(epochs = epochs, start = iter+1, lr = lr)
            self.save()

            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=int, default="1", help='number of epochs per child')
    parser.add_argument('--iterations', "-i", type=int, default="50", help='number of generations')
    parser.add_argument('--lr', "-l", type=float, default="5e-4", help='learning rate')
    parser.add_argument('--allkids', action='store_true', default=False, help='expand children tree')
    parser.add_argument('--baselines', action='store_true', default=False, help='measure baselines')
    parser.add_argument('--reg', action='store_true', default=False, help='reg group convs')
    parser.add_argument('--skip', action='store_true', default=False, help='use model with skips')
    parser.add_argument('--data', "-d", type=pathlib.Path, default="..", help='datapath')
    args = parser.parse_args()
    print(args)
    hillclimb = HillClimber(allkids=args.allkids, reg=args.reg, skip=args.skip, baselines=args.baselines, lr=args.lr, datapath=args.data)
    hillclimb.saveargs(vars(args))
    if args.baselines:
        hillclimb.baselines(iterations=args.iterations, epochs=args.epochs, lr=args.lr)
    else:
        hillclimb.hillclimb(iterations=args.iterations, epochs=args.epochs, lr=args.lr)
