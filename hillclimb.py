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
    def __init__(self, reg = False, baselines = False, pareto = False, lr = 0.1, pareto2 = False,
                 path = "..", d16 = False, c4 = False, popsize = 10, seed = -1, dea = False, skip = False,
                 test = False, folder = "", name = "", task = "mnist", unique = False, train_vanilla = False,
                 val_vanilla = False, test_vanilla = False):
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
        self.filename = str(path) +'/equiNAS/out'+folder+'/logs'+exp+'_'+name+'.pkl'
        print(self.filename)
        self.ordered = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.stagedepth = 4
        self.skip = skip
        self.lr = lr
        if "mnist" in task:
            self.train_loader, self.validation_loader, self.test_loader = utilities.get_mnist_dataloaders(path_to_dir=path, train_rot=not train_vanilla, val_rot=not val_vanilla, test_rot=not test_vanilla, 
                                                                                                          train_ood = "T" in task, val_ood = "V" in task, test_ood = "E" in task, group=task[-2:])
            self.indim = 1
            self.outdim = 10
            self.kernel = 5
            self.stages = 2
            self.pools = self.stages*2
            self.hidden = 64
            self.basechannels = 16
            dim = 28
        elif task == "isic":
            self.train_loader, self.validation_loader, self.test_loader = utilities.get_isic_dataloaders(path_to_dir=path)
            self.indim = 3
            self.outdim = 9
            self.kernel = 7
            self.stages = 2
            self.pools = 6
            self.hidden = 128
            self.basechannels = 16
            dim = 256
        elif task == "isicsmall":
            self.train_loader, self.validation_loader, self.test_loader = utilities.get_isic_dataloaders(path_to_dir=path, small=True)
            self.indim = 3
            self.outdim = 9
            self.kernel = 5
            self.stages = 2
            self.pools = 5
            self.hidden = 128
            self.basechannels = 16
            dim = 64
        elif task == "galaxy10":
            self.train_loader, self.validation_loader, self.test_loader = utilities.get_galaxy10_dataloaders(path_to_dir=path, batch_size = 32, small = False)
            self.indim = 3
            self.outdim = 10
            self.kernel = 5
            self.stages = 2
            self.pools = 6
            self.hidden = 128
            self.basechannels = 16
            dim = 256
        elif task == "galaxy10small":
            self.train_loader, self.validation_loader, self.test_loader = utilities.get_galaxy10_dataloaders(path_to_dir=path, batch_size = 64, small = True)
            self.indim = 3
            self.outdim = 10
            self.kernel = 5
            self.stages = 2
            self.pools = 5
            self.hidden = 128
            self.basechannels = 16
            dim = 64
        self.reg = reg
        if d16:
            self.g = (1,4)
        elif c4:
            self.g = (0,2)
        else:
            self.g = (1,2)
        if dea:
            model = models.DEANASNet(superspace=self.g, discrete=True, alphalr=lr, weightlr=lr, 
                                     skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                     kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels)
            x = torch.zeros(2, self.indim, dim, dim)
            for i, block in enumerate(model.blocks):
                x = block(x)
                print(x.shape)
                if i == len(model.blocks)-3:
                    x = x.reshape(x.shape[0], -1)
        else:
            model = models.SkipEquiCNN(gs=[self.g for _ in range(8)], ordered = self.ordered, lr = lr, superspace = self.g)
        self.options = [model]
        self.allkids = popsize < 0
        self.popsize = popsize
        self.pareto = pareto
        self.pareto2 = pareto2
        self.test = test
        self.unique = unique
        self.history = {}

    def train(self, epochs = 1, start = 0):
        for model in self.options:
            model = model.to(self.device)
            dataloaders = {
                "train": self.train_loader,
                "validation": self.validation_loader
            }
            print(model.countparams())
            print(torch.cuda.mem_get_info())
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
                                            'ghistory': [],
                                            'paramcounts': [],
                                            #'distances': [],
                                            'name': model.name}
                else:
                    self.history[model.uuid] = copy.deepcopy(self.history[model.parent])
            self.history[model.uuid]["epochsteps"] += np.linspace(start, start+1, int(np.ceil(epochs)), endpoint=False).tolist()
            self.history[model.uuid]["ghistory"].append(model.gs)
            self.history[model.uuid]["paramcounts"].append(model.countparams())
            #self.history[model.uuid]["distances"].append(model.distance().item())
            print(torch.cuda.mem_get_info())
            counter = 0
            for epoch in range(int(np.ceil(epochs))):
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
                        self.history[model.uuid]["trainsteps"] += [b / running_count + start + epoch for b in batch]
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

    def run_test(self, model):
        model = model.to(self.device)
        model.eval()
        running_corrects = 0
        running_count = 0
        for inputs, labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_count += inputs.size(0)
        acc = running_corrects.float() / running_count
        model = model.cpu()
        self.history[model.uuid]["test"] = acc.item()
        self.history[model.uuid]["test_distance"] = model.distance().item()

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
        if self.allkids or self.unique:
            bests = {}
            for child in self.options:
                if str(child.gs) not in bests or child.score > bests[str(child.gs)]["score"]:
                    bests[str(child.gs)] = {"uuid": child.uuid, "score": child.score}
            uuids = [bests[g]["uuid"] for g in bests.keys()]
            print(len(self.options), len(uuids), end = " -> ")
            for child in self.options:
                if child.uuid not in uuids:
                    self.options.remove(child)
            print(len(self.options))
            if self.allkids:
                return
        if self.pareto:
            costs = np.zeros((len(self.options),2))
            for i, model in enumerate(self.options):
                costs[i,0] = 1-model.score
                costs[i,1] = model.countparams()
            pareto_inds = np.where(utilities.is_pareto_efficient(costs))[0]
            if self.pareto2:
                costs[pareto_inds, :] = np.inf
                pareto_inds2 = np.where(utilities.is_pareto_efficient(costs))[0]
                pareto_inds = np.concatenate((pareto_inds, pareto_inds2))
            elif len(pareto_inds) < self.popsize:
                costs[pareto_inds, :] = np.inf
                costs = costs[:,0]
                if self.popsize>=len(costs):
                    pareto_inds = np.arange(len(costs))
                else:
                    pareto_inds2 = np.argpartition(costs, self.popsize-len(pareto_inds))[:self.popsize-len(pareto_inds)]
                    pareto_inds = np.concatenate((pareto_inds, pareto_inds2))
            for removed in [self.options[ind] for ind in range(len(self.options)) if ind not in pareto_inds]:
                self.run_test(removed)
            self.options = [self.options[ind] for ind in pareto_inds]
            print("Pareto front:", len(self.options))
            for child in sorted(self.options, key=attrgetter('score'), reverse=True):
                print(child.gs, child.countparams(), child.score)
        else:
            sorted_options = sorted(self.options, key=attrgetter('score'), reverse=True)
            self.options = sorted_options[:min(len(self.options),self.popsize)]
            for removed in sorted_options[min(len(self.options),self.popsize):]:
                self.run_test(removed)

    def save(self, end = False):
        if end and self.test:
            for child in self.options:
                if "test" not in self.history[child.uuid]:
                    self.run_test(child)
                print(child.gs, child.countparams(), self.history[child.uuid]["test"])
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)

    def hillclimb(self, generations = -1, epochs = 5.0):
        self.train(epochs = epochs, start = 0)
        for generation in range(generations):
            self.generate()
            print("Generation ", generation)
            self.train(epochs = epochs, start = generation+1)
            if generation == generations-1:
                continue
            self.select()
            self.save()
        if self.test:
            for model in self.options:
                self.run_test(model)
                print(model.gs, model.score, self.history[model.uuid]["test"])
            self.save(end=True)


    def baselines(self, generations = -1, epochs = 5.0):
        self.options[0].name = "D4 (prior: D4)"
        self.options.append(models.DEANASNet(name = "C4 (prior: C4)", superspace=(0,2), discrete=True, alphalr=self.lr, weightlr=self.lr,
                                             skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                             kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels*2))
        self.options.append(models.DEANASNet(name = "C1 (prior: C1)", superspace=(0,0), discrete=True, alphalr=self.lr, weightlr=self.lr,
                                             skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                             kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels*8))
        D4priorC1 = models.DEANASNet(name = "C1 (prior: D4)", superspace=(1,2), discrete=True, alphalr=self.lr, weightlr=self.lr,
                                     skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                     kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels)
        D4priorC4 = models.DEANASNet(name = "C4 (prior: D4)", superspace=(1,2), discrete=True, alphalr=self.lr, weightlr=self.lr,
                                     skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                     kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels)
        C4priorC1 = models.DEANASNet(name = "C1 (prior: C4)", superspace=(0,2), discrete=True, alphalr=self.lr, weightlr=self.lr,
                                     skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                     kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels*2)
        for i in range(len(D4priorC1.channels)):
                D4priorC1 = D4priorC1.offspring(len(D4priorC1.channels)-1-i, (0,0))
                D4priorC4 = D4priorC4.offspring(len(D4priorC4.channels)-1-i, (0,2))
                C4priorC1 = C4priorC1.offspring(len(C4priorC1.channels)-1-i, (0,0))
        self.options.append(D4priorC1)
        self.options.append(D4priorC4)
        self.options.append(C4priorC1)
        self.options.append(models.DEANASNet(name = "RPP D4", superspace=(1,2), discrete=True, alphalr=self.lr, weightlr=self.lr, reg_conv = 1e-6,
                                             skip=self.skip, hidden=self.hidden, indim=self.indim, outdim=self.outdim, stagedepth=self.stagedepth,
                                             kernel=self.kernel, stages=self.stages, pools=self.pools, basechannels=self.basechannels))
        self.train(epochs = epochs, start = 0)
        for generation in range(generations):
            print("Generation ", generation)
            self.train(epochs = epochs, start = generation+1)
            self.save()
        if self.test:
            for model in self.options:
                self.run_test(model)
        self.save(end=True)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=float, default="0.2", help='number of epochs per child')
    parser.add_argument('--generations', "-i", type=int, default="50", help='number of generations')
    parser.add_argument('--lr', "-l", type=float, default="0.05", help='learning rate')
    parser.add_argument('--popsize', "-p", type=int, default="5", help='population size (if not pareto)')
    parser.add_argument('--baselines', action='store_true', default=False, help='measure baselines')
    parser.add_argument('--data', "-d", type=pathlib.Path, default="..", help='datapath')
    parser.add_argument('--d16', action='store_true', default=False, help='use d16 equivariance instead of default d4')
    parser.add_argument('--c4', action='store_true', default=False, help='use c4 equivariance instead of default d4')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for unseeded)')
    parser.add_argument('--dea', action='store_true', default=False, help='use DEANAS backbone')
    parser.add_argument('--skip', action='store_true', default=False, help='turn on skip connections')
    parser.add_argument('--pareto', action='store_true', default=False, help='use pareto front as parent selection')
    parser.add_argument('--pareto2', action='store_true', default=False, help='add 2nd pareto front to parent selection')
    parser.add_argument('--unique', action='store_true', default=False, help='select unique architectures before selection')
    parser.add_argument('--test', action='store_true', default=False, help='evaluate on test set') 
    parser.add_argument('--folder', "-f", type=str, default="", help='folder to store results')
    parser.add_argument('--name', "-n", type=str, default="test", help='name of experiment')
    parser.add_argument('--task', "-t", type=str, default="mnist", help='task')
    parser.add_argument('--train_vanilla', action='store_true', default=False, help='train on vanilla data')
    parser.add_argument('--val_vanilla', action='store_true', default=False, help='val on vanilla data')
    parser.add_argument('--test_vanilla', action='store_true', default=False, help='test on vanilla data')
    args = parser.parse_args()
    if args.task == "mixmnist":
        args.train_vanilla = True
    elif "vanillamnist" in args.task:
        args.train_vanilla = True
        args.val_vanilla = True
        args.test_vanilla = True
    print(args)
    hillclimb = HillClimber(baselines=args.baselines, lr=args.lr, path=args.data, popsize=args.popsize, 
                            d16=args.d16, c4=args.c4, dea=args.dea, seed=args.seed, pareto=args.pareto, 
                            skip=args.skip, test=args.test, folder=args.folder, name=args.name, 
                            task=args.task, unique=args.unique, train_vanilla=args.train_vanilla,
                            val_vanilla=args.val_vanilla, test_vanilla=args.test_vanilla, pareto2=args.pareto2)
    hillclimb.saveargs(vars(args))
    if args.baselines:
        hillclimb.baselines(generations=args.generations, epochs=args.epochs)
    else:
        hillclimb.hillclimb(generations=args.generations, epochs=args.epochs)
