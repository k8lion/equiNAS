import datetime
import pickle
import torch

import models
import utilities

def train(model, train_loader, validation_loader, loss_function, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) #, weight_decay=1e-5)

    dataloaders = {
        "train": train_loader,
        "validation": validation_loader
    }
    save = {'train': {'loss': [], 'accuracy': [], 'naswot': [], 'batch': [], 'batchloss': []},
            'validation' : {'loss': [], 'accuracy': []}}

    for epoch in range(epochs):
        for phase in ['train', 'validation']:
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
                    save[phase]['batchloss'].append(loss.detach())
                    if hasattr(model, "K"):
                        s, ld = np.linalg.slogdet(model.K)
                        rs = np.tril(model.K/model.K[1,1], -1).sum() / (np.shape(model.K)[0]*(np.shape(model.K)[0]-1))
                        #print(relusep,ld)                        
                        save[phase]['naswot_ld'].append(ld)
                        save[phase]['naswot_rs'].append(rs)

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.float() / running_count

            save[phase]['loss'].append(epoch_loss.item())
            save[phase]['accuracy'].append(epoch_acc.item())

            if phase == "train":
                save[phase]['batch'].append([b / running_count for b in batch])
    return save

if __name__ == "__main__":
    e = datetime.datetime.now()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = 50
    train_loader, validation_loader, test_loader = utilities.get_dataloaders(path_to_dir="..")
    models = {'steerable': models.C8SteerableCNN(), 'unsteerable': models.UnsteerableCNN()}
    logs = {}
    for (name, model) in models.items():
        print(name, utilities.getmodelsize(model))
        save = train(model.to(device), train_loader, validation_loader, loss_function, epochs, device)
        logs[name] = save
    with open('./out/logs'+e.strftime("%Y-%m-%d_%H:%M:%S")+'.pkl', 'wb') as f:
        pickle.dump(logs, f)

