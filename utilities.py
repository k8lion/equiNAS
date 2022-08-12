import numpy as np
import os
import os.path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode

class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None, path_to_dir='~'):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = str(path_to_dir)+"/data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = str(path_to_dir)+"/data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
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
    

class ISICDataset(Dataset):
    """ISIC dataset."""
    # classes = {'NV': 0, 'MEL': 1, 'BKL': 2, 'DF': 3, 'SCC': 4, 'BCC': 5, 'VASC': 6, 'AK': 7}
    
    def __init__(self, data_df, path_to_dir, split='train', input_size=256, run_id=1):
        """
        Args:
        """
        self.split = split
        self.input_size = input_size
        self.data_df = data_df
        self.path_to_dir = path_to_dir
        # if split == "train":
        #     self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                  transforms.RandomVerticalFlip(),
        #                                  transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        #                                  transforms.Resize(self.input_size),
        #                                  transforms.ToTensor()])
        # elif split == "val":
        self.trans = Compose([Resize(self.input_size),
                              ToTensor()])
        
    
    def __getitem__(self, idx):
        val_start_id = int(len(self.data_df) * 0.8)
        if self.split == "train":
            idx = idx
        elif self.split == "val":
            idx = idx + val_start_id
        image_id = self.data_df.loc[idx, "image"]
        y = self.data_df.loc[idx, "target"]
        x = Image.open(os.path.join(self.path_to_dir, "/data/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input", "{}.jpg".format(image_id)))
        x = self.center_crop(x)
        x = self.trans(x)
        y = np.int64(y)
        return {"image": x, "label": y}
            
    def __len__(self):
        if self.split == "train":
            return int(len(self.data_df) * 0.8)
        elif self.split == "val":
            return int(len(self.data_df) * 0.2)
    
    def center_crop(self, pil_img):
        img_width, img_height = pil_img.size
        if img_width > img_height:
            crop_size = img_height
        else:
            crop_size = img_width
        return pil_img.crop(((img_width - crop_size) // 2,
                             (img_height - crop_size) // 2,
                             (img_width + crop_size) // 2,
                             (img_height + crop_size) // 2))


def get_mnist_dataloaders(path_to_dir = "~", validation_split=0.2, batch_size=64):
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
    test_transform = Compose([
        pad,
        totensor,
    ])

    mnist_train = MnistRotDataset(mode='train', transform=test_transform, path_to_dir=path_to_dir)

    shuffle_dataset = True
    random_seed = 42
    dataset_size = mnist_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(mnist_train, batch_size=batch_size,
                                                    sampler=valid_sampler)

    mnist_test = MnistRotDataset(mode='test', transform=test_transform, path_to_dir=path_to_dir)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader

def get_isic_dataloaders(path_to_dir = "~", validation_split=0.2, batch_size=64):
    groundtruth_df = pd.read_csv(str(path_to_dir)+"/data/isic-2019/ISIC_2019_Training_GroundTruth.csv")
    groundtruth_df["target"] = groundtruth_df.apply(lambda x: np.argmax(x[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]]), axis=1)
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

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