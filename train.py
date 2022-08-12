import datetime
import pickle
import torch
import numpy as np
import models
import utilities

def train(model, train_loader, validation_loader, loss_function, epochs, device):
    if hasattr(model, "alphas"):
        optimizer = torch.optim.Adam(list(model.parameters())+[model.alphas], lr=5e-5) #, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) #, weight_decay=1e-5)

    dataloaders = {
        "train": train_loader,
        "validation": validation_loader
    }
    save = {'train': {'loss': [], 
                      'accuracy': [], 
                      'naswot_ld': [], 
                      'naswot_rs': [], 
                      'naswot_ldU': [], 
                      'naswot_rsU': [], 
                      'naswot_ldS': [], 
                      'naswot_rsS': [], 
                      'naswot_ldF': [], 
                      'naswot_rsF': [], 
                      'alphas': [[] for _ in range(10)],
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
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_count += inputs.size(0)
                if phase == "train":
                    batch.append(running_count)
                    save[phase]['batchloss'].append(loss.detach().item())
                    if hasattr(model, "K"):
                        s, ld = np.linalg.slogdet(model.K)
                        rs = np.tril(model.K/model.K[1,1], -1).sum() / (np.shape(model.K)[0]*(np.shape(model.K)[0]-1))
                        save[phase]['naswot_ld'].append(ld)
                        save[phase]['naswot_rs'].append(rs)
                    if hasattr(model, "KU"):
                        s, ld = np.linalg.slogdet(model.KU)
                        rs = np.tril(model.KU/model.KU[1,1], -1).sum() / (np.shape(model.KU)[0]*(np.shape(model.K)[0]-1))
                        save[phase]['naswot_ldU'].append(ld)
                        save[phase]['naswot_rsU'].append(rs)
                    if hasattr(model, "KS"):
                        s, ld = np.linalg.slogdet(model.KS)
                        rs = np.tril(model.KS/model.KS[1,1], -1).sum() / (np.shape(model.KS)[0]*(np.shape(model.K)[0]-1))
                        save[phase]['naswot_ldS'].append(ld)
                        save[phase]['naswot_rsS'].append(rs)
                    if hasattr(model, "KF"):
                        s, ld = np.linalg.slogdet(model.KF)
                        rs = np.tril(model.KF/model.KF[1,1], -1).sum() / (np.shape(model.KF)[0]*(np.shape(model.K)[0]-1))
                        save[phase]['naswot_ldF'].append(ld)
                        save[phase]['naswot_rsF'].append(rs)
                    if hasattr(model, "alphas"):
                        for i in range(10):
                            save[phase]['alphas'][i].append(model.alphas[i].item())

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.float() / running_count

            save[phase]['loss'].append(epoch_loss.item())
            save[phase]['accuracy'].append(epoch_acc.item())

            print("{0:.3g} ".format(epoch_loss.item()), end="")
            print("{0:.3g} ".format(epoch_acc.item()), end="\t")

            if phase == "train":
                save[phase]['batch'].append([b / running_count + epoch for b in batch])
            else:
                print("")

    return save

if __name__ == "__main__":
    e = datetime.datetime.now()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = 50
    train_loader, validation_loader, test_loader = utilities.get_mnist_dataloaders(path_to_dir="..")
    #models = {'steerable': models.C8SteerableCNN(), 'unsteerable': models.UnsteerableCNN()}
    models = {'soft': models.C8MutantCNN(soft=True), 'hard': models.C8MutantCNN(soft=False)}
    logs = {}
    for (name, model) in models.items():
        print(name, utilities.getmodelsize(model))
        save = train(model.to(device), train_loader, validation_loader, loss_function, epochs, device)
        logs[name] = save
    with open('./out/logsmutant'+e.strftime("%Y-%m-%d_%H:%M:%S")+'.pkl', 'wb') as f:
        pickle.dump(logs, f)

