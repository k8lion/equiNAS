import datetime
import pickle
import numpy as np
import models
import utilities
from operator import attrgetter
import argparse
import pathlib
import torch
            
def DEANASearch(args):
    filename = str(args.path) +'/equiNAS/out/logsdea_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loader, validation_loader, test_loader = utilities.get_dataloaders(path_to_dir=args.path, validation_split=0.5)
    history = {"args": args,
                'train': {'loss': [], 
                        'accuracy': [], 
                        'batch': [], 
                        'batchloss': []},
                'trainsteps': [],
                'validation' : {'loss': [], 
                                'accuracy': []},
                'epochsteps': []}
    model = models.DEANASNet().to(device)
    for epoch in range(args.epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.optimizer.zero_grad()
                model.alphaopt.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch_idx % args.log_interval == 0:
                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        phase, epoch, batch_idx * len(inputs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[phase]['loss'].append(epoch_loss)
            history[phase]['accuracy'].append(epoch_acc)
            history[phase]['batch'].append(batch_idx)
            history[phase]['batchloss'].append(loss.item())
            history[phasesteps].append(epoch)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=int, default="1", help='number of epochs per child')
    #parser.add_argument('--lr', "-l", type=float, default="5e-4", help='learning rate')
    parser.add_argument('--data', "-d", type=pathlib.Path, default="..", help='datapath')
    args = parser.parse_args()
    DEANASearch(args)
    