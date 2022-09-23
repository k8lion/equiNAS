import datetime
import pickle
import numpy as np
import models
import utilities
import argparse
import pathlib
import os
"""
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
#from ray.tune.search.optuna import OptunaSearch

def DEANASearch_tune(args):
    if str(args.path)[0] != '/':
        args.path = os.getcwd() / args.path
    config = {
        "hidden": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        "alphalr": tune.choice([1e-3, 3e-3, 5e-3, 1e-2, 3e-2]),
        "weightlr": tune.choice([1e-3, 3e-3, 5e-3, 1e-2, 3e-2]),
        "basechannels": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
    }
    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    #algo = OptunaSearch()
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(DEANASearch_train, args=args), {"cpu": 2, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=1000,
            #search_alg = algo,
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
    args.basechannels = config["basechannels"]
    print(config)
    DEANASearch(args)
"""
def DEANASearch(args):
    import torch
    trial = "/logsdea_"
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        trial += str(args.seed) + "_"
    if args.baseline:
        trial += "bl"
        if not args.prior:
            trial+="C1"
        else:
            if args.c4:
                trial+="C4"
            elif args.d16:
                trial+="D16"
            else:
                trial+="D4"
        if not args.skip:
            trial+="ns"
        trial+="_"
    else:
        trial+="dea"
        if not args.prior:
            trial+="eq"
        if not args.skip:
            trial+="ns"
        trial+="_"
    filename = str(args.path) +'/equiNAS/out'+args.folder+trial+args.name+'.pkl'
    print(filename)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(args.task)
    if "mnist" in args.task:
        train_loader, validation_loader, test_loader = utilities.get_mnist_dataloaders(path_to_dir=args.path, batch_size=args.batch_size, train_rot=not args.train_vanilla, val_rot=not args.val_vanilla, test_rot=not args.test_vanilla)
        args.indim = 1
        args.outdim = 10
        if args.kernel < 0:
            args.kernel = 5
        args.stages = 2
        args.pools = args.stages*2
        if args.hidden < 0:
            args.hidden = 64
        if args.basechannels < 0:
            args.basechannels = 16
        dim = 28
    elif args.task == "isic":
        train_loader, validation_loader, test_loader = utilities.get_isic_dataloaders(path_to_dir=args.path, batch_size=args.batch_size)
        args.indim = 3
        args.outdim = 9
        if args.kernel < 0:
            args.kernel = 7
        args.stages = 2
        args.pools = 6
        if args.hidden < 0:
            args.hidden = 128
        if args.basechannels < 0:
            args.basechannels = 16
        dim = 256
    elif args.task == "galaxy10":
        train_loader, validation_loader, test_loader = utilities.get_galaxy10_dataloaders(path_to_dir=args.path, batch_size=args.batch_size)
        args.indim = 3
        args.outdim = 10
        if args.kernel < 0:
            args.kernel = 7
        args.stages = 2
        args.pools = 6
        if args.hidden < 0:
            args.hidden = 128
        if args.basechannels < 0:
            args.basechannels = 16
        dim = 256
    if args.baseline and args.c4:
        args.basechannels *= 2
    model = models.DEANASNet(superspace = (1,4) if args.d16 else (0,2) if args.c4 else (1,2), hidden = args.hidden,
                             weightlr = args.weightlr, alphalr = args.alphalr, basechannels = args.basechannels,
                             prior = args.prior, indim = args.indim, baseline = args.baseline,
                             outdim = args.outdim, stages = args.stages, pools = args.pools, 
                             kernel = args.kernel, skip = args.skip).to(device)
    x = torch.zeros(2, args.indim, dim, dim)
    for i, block in enumerate(model.blocks):
        x = block(x)
        print(x.shape)
        if i == len(model.blocks)-3:
            x = x.reshape(x.shape[0], -1)
    history = {'args': args,
                'alphas': [[torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()]],
                'channels': model.channels,
                'groups': model.groups, 
                'train': {'loss': [], 
                        'accuracy': [], 
                        'batchloss': []},
                'trainsteps': [],
                'validation' : {'loss': [], 
                                'accuracy': []},
                'distances': [model.distance(layerwise=True)],
    }
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    model.optimizer, float(args.epochs), eta_min=1e-4)
    scheduler = None
    for epoch in range(args.epochs):
        print(torch.softmax(model.blocks[1]._modules["0"].alphas, dim=0))
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
                    if not args.tune:
                        history['train']['batchloss'].append(loss.detach().item())
            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects / running_count
            if not args.tune:
                print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
                history[phase]['loss'].append(epoch_loss)
                history[phase]['accuracy'].append(epoch_acc)
                if phase == 'train':
                    history["trainsteps"] += [epoch + b / running_count for b in batch]
            elif phase == "validation":
                session.report({"loss": epoch_loss, "accuracy": epoch_acc})
        history["distances"].append(model.distance(layerwise=True))
        if not args.tune:
            history['alphas'].append([torch.softmax(a, dim=0).detach().tolist() for a in model.alphas()])
        if scheduler is not None:
            scheduler.step()
    if args.test:
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        running_count = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = model.loss_function(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_count += inputs.size(0)
        epoch_loss = running_loss / running_count
        epoch_acc = running_corrects / running_count
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        history['test'] = {'loss': epoch_loss, 'accuracy': epoch_acc, 'distance': model.distance().item()}
    if not args.tune:
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
    parser.add_argument('--prior', action='store_true', default=False, help='do not equalize initial alphas')
    parser.add_argument('--skip', action='store_true', default=False, help='turn on skip connections') 
    #RPP
    parser.add_argument('--baseline', action='store_true', default=False, help='lock network to C1+skip')
    parser.add_argument('--task', "-t", type=str, default="mnist", help='task')
    parser.add_argument('--seed', "-s", type=int, default=-1, help='random seed (-1 for unseeded)')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size (-1 is default for given task)')
    parser.add_argument('--d16', action='store_true', default=False, help='use d16 equivariance instead of default d4')
    parser.add_argument('--c4', action='store_true', default=False, help='use c4 equivariance instead of default d4') 
    parser.add_argument('--tune', action='store_true', default=False, help='tune hyperparameters') 
    parser.add_argument('--test', action='store_true', default=False, help='evaluate on test set') 
    parser.add_argument('--folder', "-f", type=str, default="", help='folder to store results')
    parser.add_argument('--name', "-n", type=str, default="test", help='name of experiment')
    parser.add_argument('--train_vanilla', action='store_true', default=False, help='train on vanilla data')
    parser.add_argument('--val_vanilla', action='store_true', default=False, help='val on vanilla data')
    parser.add_argument('--test_vanilla', action='store_true', default=False, help='test on vanilla data')
    args = parser.parse_args()
    if args.task == "mixmnist":
        args.train_vanilla = True
    print(args)
    if args.tune:
        DEANASearch_tune(args)
    else:
        DEANASearch(args)
    
