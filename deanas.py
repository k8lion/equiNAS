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
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    filename = str(args.path) +'/equiNAS/out/logsdea_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
    print(filename)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.task == "mnist":
        train_loader, validation_loader, test_loader = utilities.get_mnist_dataloaders(path_to_dir=args.path, validation_split=0.5)
    model = models.DEANASNet(superspace = (1,4) if args.d16 else (1,2), weightlr = args.weightlr, alphalr = args.alphalr, prior = not args.equalize).to(device)
    history = {'args': args,
                'alphas': [],
                'channels': model.channels,
                'groups': model.groups, 
                'train': {'loss': [], 
                        'accuracy': [], 
                        'batchloss': []},
                'trainsteps': [],
                'validation' : {'loss': [], 
                                'accuracy': []},
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, float(args.epochs), eta_min=1e-4)
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
                running_corrects += torch.sum(preds == labels.data).item()
                running_count += inputs.size(0)
                if phase == 'train':
                    batch.append(running_count)
                    history['train']['batchloss'].append(loss.detach().item())
            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects / running_count
            print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            history[phase]['loss'].append(epoch_loss)
            history[phase]['accuracy'].append(epoch_acc)
            if phase == 'train':
                history["trainsteps"] += [epoch + b / running_count for b in batch]
        history['alphas'].append([torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()])
        scheduler.step()

    
    with open(filename, 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run diffentiable equivariance-aware NAS')
    parser.add_argument('--epochs', "-e", type=int, default="50", help='number of epochs per child')
    parser.add_argument('--weightlr', "-w", type=float, default="1e-3", help='weight learning rate')
    parser.add_argument('--alphalr', "-a", type=float, default="1e-3", help='alpha learning rate')
    parser.add_argument('--path', "-p", type=pathlib.Path, default="..", help='datapath')
    parser.add_argument('--equalize', action='store_true', default=False, help='eqaulize initial alphas')
    parser.add_argument('--task', "-t", type=str, default="mnist", help='task')
    parser.add_argument('--seed', "-s", type=int, default=-1, help='random seed (-1 for unseeded)')
    parser.add_argument('--d16', action='store_true', default=False, help='use d16 equivariance instead of default d4')
    args = parser.parse_args()
    print(args)
    DEANASearch(args)
    