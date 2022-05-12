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
    save = {'train': {'loss': [], 'accuracy': []},
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

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.float() / running_count

            save[phase]['loss'].append(epoch_loss.item())
            save[phase]['accuracy'].append(epoch_acc.item())
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
        save = train(model, train_loader, validation_loader, loss_function, epochs, device)
        logs[name] = save
    with open('./out/logs'+e.strftime("%Y-%m-%d_%H:%M:%S")+'.pkl', 'wb') as f:
        pickle.dump(logs, f)

