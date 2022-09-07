import datetime
import pickle
import numpy as np
import models
import utilities
import argparse
import pathlib
import os

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler


def DEANASearch_tune(args):
    if str(args.path)[0] != '/':
        args.path = os.getcwd() / args.path
    config = {
        "hidden": tune.sample_from(lambda _: 2 ** np.random.randint(5, 10)),
        "alphalr": tune.choice([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
        "weightlr": tune.choice([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
        "basechannels": tune.sample_from(lambda _: 2 ** np.random.randint(3, 8)),
        "kernel": tune.choice([3, 5, 7, 9]),
    }
    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(DEANASearch_train, args=args),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
        
def DEANASearch_train(config, args):
    args.hidden = config["hidden"]
    args.alphalr = config["alphalr"]
    args.weightlr = config["weightlr"]
    args.batch_size = config["batch_size"]
    args.basechannels = config["basechannels"]
    args.kernel = config["kernel"]
    print(config)
    DEANASearch(args)

def DEANASearch(args):
    import torch
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
    print(args.task)
    if args.task == "mnist":
        train_loader, validation_loader, _ = utilities.get_mnist_dataloaders(path_to_dir=args.path, validation_split=0.5, batch_size=args.batch_size)
        args.indim = 1
        args.outdim = 10
        args.pools = 4
        if args.kernel < 0:
            args.kernel = 5
        args.stages = 2
        if args.hidden < 0:
            args.hidden = 64
        if args.basechannels < 0:
            args.basechannels = 16
    elif args.task == "isic":
        train_loader, validation_loader, _ = utilities.get_isic_dataloaders(path_to_dir=args.path, validation_split=0.5, batch_size=args.batch_size)
        args.indim = 3
        args.outdim = 9
        args.pools = 8
        if args.kernel < 0:
            args.kernel = 7
        args.stages = 4
        if args.hidden < 0:
            args.hidden = 256
        if args.basechannels < 0:
            args.basechannels = 64
    elif args.task == "galaxy10":
        train_loader, validation_loader, _ = utilities.get_galaxy10_dataloaders(path_to_dir=args.path, validation_split=0.5, batch_size=args.batch_size)
        args.indim = 3
        args.outdim = 10
        args.pools = 8
        if args.kernel < 0:
            args.kernel = 7
        args.stages = 4
        if args.hidden < 0:
            args.hidden = 128
        if args.basechannels < 0:
            args.basechannels = 32
    model = models.DEANASNet(superspace = (1,4) if args.d16 else (0,2) if args.c4 else (1,2), hidden = args.hidden,
                             weightlr = args.weightlr, alphalr = args.alphalr, basechannels = args.basechannels,
                             prior = not args.equalize, indim = args.indim, baseline = args.baseline,
                             outdim = args.outdim, stages = args.stages, pools = args.pools, 
                             kernel = args.kernel).to(device)
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
            if tune and phase == 'validation' and epoch < args.epochs-1:
                continue
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
                        if not args.baseline:
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
            elif tune:
                session.report({"loss": epoch_loss, "accuracy": epoch_acc})
        history['alphas'].append([torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()])
        scheduler.step()

    with open(filename, 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run diffentiable equivariance-aware NAS')
    parser.add_argument('--epochs', "-e", type=int, default="50", help='number of epochs per child')
    parser.add_argument('--weightlr', "-w", type=float, default="1e-3", help='weight learning rate')
    parser.add_argument('--alphalr', "-a", type=float, default="1e-3", help='alpha learning rate')
    parser.add_argument('--hidden', "-b", type=int, default="-1", help='width of fully connected layer')
    parser.add_argument('--basechannels', "-c", type=int, default="-1", help='base number of channels')
    parser.add_argument('--kernel', "-k", type=int, default="-1", help='kernel size')
    parser.add_argument('--path', "-p", type=pathlib.Path, default="..", help='datapath')
    parser.add_argument('--equalize', action='store_true', default=False, help='eqaulize initial alphas')
    parser.add_argument('--baseline', action='store_true', default=False, help='lock network to C1+skip')
    parser.add_argument('--task', "-t", type=str, default="mnist", help='task')
    parser.add_argument('--seed', "-s", type=int, default=-1, help='random seed (-1 for unseeded)')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size (-1 is default for given task)')
    parser.add_argument('--d16', action='store_true', default=False, help='use d16 equivariance instead of default d4')
    parser.add_argument('--c4', action='store_true', default=False, help='use c4 equivariance instead of default d4') 
    parser.add_argument('--tune', action='store_true', default=False, help='tune hyperparameters') 
    args = parser.parse_args()
    print(args)
    if args.tune:
        DEANASearch_tune(args)
    else:
        DEANASearch(args)
    