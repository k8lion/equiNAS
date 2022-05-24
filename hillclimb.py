import datetime
import pickle
import torch
import numpy as np
import models
import utilities
from operator import attrgetter

class HillClimber(object):
    def __init__(self, reset = True):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.validation_loader, self.test_loader = utilities.get_dataloaders(path_to_dir="..")
        self.model = models.EquiCNN(reset)
        self.options = []

    def train(self, epochs = 1):
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
            save = {'train': {'loss': [], 
                            'accuracy': [], 
                            'batch': [], 
                            'batchloss': []},
                    'validation' : {'loss': [], 'accuracy': []}}

            for epoch in range(epochs):
                print(epoch, end="\t")
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
                            save[phase]['batchloss'].append(loss.detach().item())

                    epoch_loss = running_loss / running_count
                    epoch_acc = running_corrects.float() / running_count

                    save[phase]['loss'].append(epoch_loss.item())
                    save[phase]['accuracy'].append(epoch_acc.item())
                    model.score = epoch_acc.item()

                    print("{0:.3g} ".format(epoch_loss.item()), end="")
                    print("{0:.3g} ".format(epoch_acc.item()), end="\t")

                    if phase == "train":
                        save[phase]['batch'].append([b / running_count + epoch for b in batch])
                    else:
                        print("")
            print(model.gs, model.score)
            model = model.to("cpu")
        #return save
    
    def generate(self):
        self.options = self.model.generate()

    def select(self):
        self.model = max(self.options, key=attrgetter("score"))
        self.options = []

    def hillclimb(self, iterations = -1):
        self.train(epochs = 10)
        while iterations > 0:
            self.generate()
            self.train(epochs = 5)
            self.select()
            iterations -= 1


if __name__ == "__main__":
    hillclimb = HillClimber()
    hillclimb.hillclimb(iterations=10)