import datetime
import pickle
import torch
import numpy as np
import models
import utilities
from operator import attrgetter
import argparse
import copy


class HillClimber(object):
    def __init__(self, reset = True, allkids = False):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.validation_loader, self.test_loader = utilities.get_dataloaders(path_to_dir="..")
        self.model = models.EquiCNN(reset)
        self.options = []
        self.allkids = allkids
        self.history = {}
        self.filename = './out/logshc_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'


    def train(self, epochs = 1, start = 0):
        if len(self.options)==0:
            totrain = [self.model]
        else:
            totrain = self.options
        for model in totrain:
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
            #self.history[model.uuid]["trainsteps"] += np.linspace(0, 1, epochs*len(dataloaders["train"]), endpoint=False).tolist()
            self.history[model.uuid]["epochsteps"] += np.linspace(start, start+1, epochs, endpoint=False).tolist()
            self.history[model.uuid]["ghistory"] += model.gs
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
            model = model.to("cpu")
    
    def generate(self):
        if self.allkids:
            children = [self.model]
            children += self.model.generate()
            for model in self.options:
                children += model.generate()
            self.options = children
        else:
            self.options = self.model.generate()

    def select(self):
        for child in self.options:
            print(child.gs, sum(p.numel() for p in child.parameters() if p.requires_grad), child.score)
        self.model = max(self.options, key=attrgetter("score"))

    def save(self):
        for model in self.options:
            self.history[model.uuid] = self.history[model.uuid]
        with open(self.filename, 'wb') as f:
            pickle.dump(self.history, f)


    def hillclimb(self, iterations = -1, epochs = 5):
        self.train(epochs = epochs, start = 0)
        for iter in range(iterations):
            self.generate()
            print("Iteration ", iter)
            self.train(epochs = epochs, start = iter)
            self.select()
            self.save()
            
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=int, default="5", help='number of epochs per child')
    parser.add_argument('--iterations', "-i", type=int, default="20", help='number of generations')
    parser.add_argument('--allkids', action='store_true', default=False, help='expand children tree')
    args = parser.parse_args()
    print(args)
    hillclimb = HillClimber(allkids=args.allkids)
    hillclimb.hillclimb(iterations=args.iterations, epochs=args.epochs)