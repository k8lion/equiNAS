import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode

class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None, path_to_dir='~'):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = path_to_dir+"/data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = path_to_dir+"/data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(os.path.expanduser(file), delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

        #self.images = np.pad(self.images, pad_width=((0,0), (2, 3), (2, 3)), mode='edge')

    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode='F')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)
    

def get_dataloaders(path_to_dir = "~"):
    pad = Pad((0, 0, 1, 1), fill=0)
    resize1 = Resize(87)
    resize2 = Resize(29)
    totensor = ToTensor()
    train_transform = Compose([
        pad,
        resize1,
        RandomRotation(180., interpolation=InterpolationMode.BILINEAR, expand=False),
        resize2,
        totensor,
    ])

    mnist_train = MnistRotDataset(mode='train', transform=train_transform, path_to_dir=path_to_dir)

    shuffle_dataset = True
    random_seed= 42
    validation_split = .2
    dataset_size = mnist_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    val_indices, train_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)
    train_loader = DataLoader(mnist_train, batch_size=64, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(mnist_train, batch_size=64,
                                                    sampler=valid_sampler)

    test_transform = Compose([
        pad,
        totensor,
    ])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform, path_to_dir=path_to_dir)
    test_loader = DataLoader(mnist_test, batch_size=64)

    return train_loader, validation_loader, test_loader

def getmodelsize(model, includebuffer=False, counts=True):
    param_size = 0
    for param in model.parameters():
        if counts:
            multiplier = 1
        else:
            multiplier = param.element_size()
        param_size += param.nelement() * multiplier
    if includebuffer:
        buffer_size = 0
        for buffer in model.buffers():
            if counts:
                multiplier = 1
            else:
                multiplier = buffer.element_size()
            buffer_size += buffer.nelement() * multiplier
        return buffer_size + param_size
    return param_size