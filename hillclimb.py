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

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1) 
            is_efficient[i] = True 
    return is_efficient

class HillClimber(object):
    def __init__(self, reset = True, reg = False, skip = False, baselines = False, pareto = False, lr = 0.1, path = "..", d16 = False, c4 = False, popsize = 10, seed = -1, dea = False):
        self.seed = seed
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if baselines:
            exp = "bs"
        else:
            exp = "hc"
        self.filename = str(path) +'/equiNAS/out/logs'+exp+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
        print(self.filename)
        self.ordered = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.validation_loader, self.test_loader = utilities.get_mnist_dataloaders(path_to_dir=path)
        self.reg = reg
        if d16:
            self.g = (1,4)
        elif c4:
            self.g = (0,2)
        else:
            self.g = (1,2)
        if dea:
            model = models.DEANASNet(superspace=self.g, discrete=True, alphalr=lr, weightlr=lr)
        else:
            model = models.SkipEquiCNN(gs=[self.g for _ in range(8)], ordered = self.ordered, lr = lr, superspace = self.g)
        self.lr = lr
        self.skip = True
        self.options = [model]
        self.allkids = popsize < 0
        self.popsize = popsize
        self.pareto = pareto
        self.history = {}

    def train(self, epochs = 1, start = 0):
        for model in self.options:
            model = model.to(self.device)
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
            self.history[model.uuid]["epochsteps"] += np.linspace(start, start+1, int(np.ceil(epochs)), endpoint=False).tolist()
            self.history[model.uuid]["ghistory"].append(model.gs)
            counter = 0
            for _ in range(int(np.ceil(epochs))):
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
                            counter += 1
                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.detach() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        running_count += inputs.size(0)
                        if phase == "train":
                            batch.append(running_count)
                            self.history[model.uuid][phase]['batchloss'].append(loss.detach().item())
                            if counter > len(dataloaders[phase])*epochs:
                                break
                    epoch_loss = running_loss / running_count
                    epoch_acc = running_corrects.float() / running_count

                    self.history[model.uuid][phase]['loss'].append(epoch_loss.item())
                    self.history[model.uuid][phase]['accuracy'].append(epoch_acc.item())
                    model.score = epoch_acc.item()

                    if phase == "train":
                        self.history[model.uuid]["trainsteps"] += [b / running_count + start for b in batch]
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

        model = model.cpu()

        return acc.item()

    def saveargs(self, args):
        self.history["args"] = args
    
    def generate(self):
        children = []
        for model in self.options:
            print("parent:", model.gs, self.validate(model))
            model = model.cpu()
            for child in model.generate():
                if any([g == (1,1) for g in child.gs]): #DELETE 
                    continue
                print("child:", child.gs, self.validate(child))
                children.append(child)
        self.options = children


    def select(self):
        for child in sorted(self.options, key=attrgetter('score'), reverse=True):
            print(child.gs, child.countparams(), child.score)
        #self.model = max(self.options, key=attrgetter("score"))
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
        else:
            if self.pareto:
                costs = np.zeros((len(self.options),2))
                for i, model in enumerate(self.options):
                    costs[i,0] = model.score
                    costs[i,1] = model.countparams()
                self.options = [self.options[ind] for ind in np.where(is_pareto_efficient(costs))[0]]
                print("Pareto front:", len(self.options))
                for child in sorted(self.options, key=attrgetter('score'), reverse=True):
                    print(child.gs, child.countparams(), child.score)
            else:
                self.options = sorted(self.options, key=attrgetter('score'), reverse=True)[:min(len(self.options),self.popsize)]

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)

    def hillclimb(self, iterations = -1, epochs = 5.0):
        self.train(epochs = epochs, start = 0)
        for iteration in range(iterations):
            self.generate()
            print("Iteration ", iteration)
            self.train(epochs = epochs, start = iteration+1)
            self.select()
            self.save()

    def baselines(self, iterations = -1, epochs = 5.0):
        self.options.append(models.DEANASNet(superspace=(0,2), discrete=True, alphalr=self.lr, weightlr=self.lr))
        self.options.append(models.DEANASNet(superspace=(0,0), discrete=True, alphalr=self.lr, weightlr=self.lr))
        self.train(epochs = epochs, start = 0)
        for iteration in range(iterations):
            print("Iteration ", iteration)
            self.train(epochs = epochs, start = iteration+1)
            self.save()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=float, default="1.0", help='number of epochs per child')
    parser.add_argument('--iterations', "-i", type=int, default="50", help='number of generations')
    parser.add_argument('--lr', "-l", type=float, default="5e-4", help='learning rate')
    parser.add_argument('--allkids', action='store_true', default=False, help='expand children tree')
    parser.add_argument('--popsize', "-p", type=int, default="10", help='population size')
    parser.add_argument('--baselines', action='store_true', default=False, help='measure baselines')
    parser.add_argument('--data', "-d", type=pathlib.Path, default="..", help='datapath')
    parser.add_argument('--d16', action='store_true', default=False, help='use d16 equivariance instead of default d4')
    parser.add_argument('--c4', action='store_true', default=False, help='use c4 equivariance instead of default d4')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for unseeded)')
    parser.add_argument('--dea', action='store_true', default=False, help='use DEANAS backbone')
    parser.add_argument('--pareto', action='store_true', default=False, help='use pareto front')
    args = parser.parse_args()
    print(args)
    hillclimb = HillClimber(baselines=args.baselines, lr=args.lr, path=args.data, popsize=args.popsize, d16=args.d16, c4=args.c4, dea=args.dea, seed=args.seed, pareto=args.pareto)
    hillclimb.saveargs(vars(args))
    if args.baselines:
        hillclimb.baselines(iterations=args.iterations, epochs=args.epochs)
    else:
        hillclimb.hillclimb(iterations=args.iterations, epochs=args.epochs)
