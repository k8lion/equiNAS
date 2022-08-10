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
    history = {'args': args,
                'alphas': [],
                'train': {'loss': [], 
                        'accuracy': [], 
                        'batchloss': []},
                'trainsteps': [],
                'validation' : {'loss': [], 
                                'accuracy': []},
    }
    model = models.DEANASNet(weightlr = args.weightlr, alphalr = args.alphalr).to(device)
    history['alphas'].append([torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()])
    for epoch in range(args.epochs):
        for phase in ['train', 'validation']:
            batch = []
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_count = 0
            loader = train_loader if phase == 'train' else validation_loader
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if phase == 'train':
                    inputs_search, labels_search = next(iter(validation_loader))
                    inputs_search = inputs_search.to(device)
                    labels_search = labels_search.to(device)
                model.optimizer.zero_grad()
                model.alphaopt.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = model.loss_function(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        model.optimizer.step()
                        outputs_search = model(inputs_search)
                        loss_search = model.loss_function(outputs_search, labels_search)
                        loss_search.backward()
                        model.alphaopt.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_count += inputs.size(0)
                if phase == 'train':
                    batch.append(running_count)
                    history['train']['batchloss'].append(loss.detach().item())
            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.double() / running_count
            print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            history[phase]['loss'].append(epoch_loss)
            history[phase]['accuracy'].append(epoch_acc)
            if phase == 'train':
                history["trainsteps"] += [b / running_count for b in batch]
        history['alphas'].append([torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()])
    
    with open(filename, 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hillclimber algorithm')
    parser.add_argument('--epochs', "-e", type=int, default="50", help='number of epochs per child')
    parser.add_argument('--weightlr', "-w", type=float, default="1e-3", help='weight learning rate')
    parser.add_argument('--alphalr', "-a", type=float, default="1e-3", help='alpha learning rate')
    parser.add_argument('--path', "-p", type=pathlib.Path, default="..", help='datapath')
    args = parser.parse_args()
    DEANASearch(args)
    