from filelock import FileLock
import h5py
import numpy as np
import os
import os.path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
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
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode='F')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

class Galaxy10Dataset(Dataset):
    def __init__(self, path_to_dir='~', transform=ToTensor(), mode="train"):
        if mode == "train":
            file = str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"
        else:
            file = str(path_to_dir)+"/data/Galaxy10_DECals_test.h5"
        
        f = h5py.File(file, 'r')
        labels, images = f['labels'], f['images']
        print(type(images))
        print(np.mean(images, axis = (0,1,2)), np.std(images, axis = (0,1,2)))

        self.x = images
        self.y = torch.from_numpy(labels[:]).long()
        self.num_samples = len(images)

        self.transform = transform

    def __getitem__(self, item):
        img = self.x[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[item]

    def __len__(self):
        return self.num_samples

class ISICDataset(Dataset):
    """ISIC dataset."""
    # classes = {'NV': 0, 'MEL': 1, 'BKL': 2, 'DF': 3, 'SCC': 4, 'BCC': 5, 'VASC': 6, 'AK': 7}
    
    def __init__(self, data_df, path_to_dir, input_size=256):

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
        image_id = self.data_df.loc[idx, "image"]
        y = self.data_df.loc[idx, "target"]
        x = Image.open(str(self.path_to_dir)+"/data/ISIC_2019/ISIC_2019_Training_Input/{}.jpg".format(image_id))
        x = self.center_crop(x)
        x = self.trans(x)
        y = np.int64(y)
        return x, y
            
    def __len__(self):
        return len(self.data_df)
    
    def get_stats(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        for i in range(len(self.data_df)):
            x, _ = self.__getitem__(i)
            mean += x.numpy().mean(axis=(1, 2))
            std += x.numpy().std(axis=(1, 2))
        mean /= len(self.data_df)
        std /= len(self.data_df)
        return mean, std
    
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
    if batch_size < 0:
        batch_size = 64
    totensor = ToTensor()
    transform = Compose([
        totensor,
        Normalize(0.12996, 0.29698),
    ])

    mnist_train = MnistRotDataset(mode='train', transform=transform, path_to_dir=path_to_dir)

    shuffle_dataset = True
    random_seed = 42
    dataset_size = mnist_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    with FileLock(os.path.expanduser("~/.data.lock")):
        train_loader = DataLoader(mnist_train, batch_size=batch_size, 
                                                sampler=train_sampler)
        validation_loader = DataLoader(mnist_train, batch_size=batch_size,
                                                        sampler=valid_sampler)

    mnist_test = MnistRotDataset(mode='test', transform=transform, path_to_dir=path_to_dir)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def get_galaxy10_dataloaders(path_to_dir = "~", validation_split=0.2, batch_size=8):
    if batch_size < 0:
        batch_size = 8
    if not os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"):
        if os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals.h5"):
            make_galaxy10_traintest(path_to_dir)
        else:
            print("No data found")
            return None, None, None
    totensor = ToTensor()
    transform = Compose([
        totensor,
        Normalize([42.70899344, 41.46352323, 40.52334547], [32.81805034, 30.10008284, 28.462431]),
    ])
    galaxy10_train = Galaxy10Dataset(mode='train', transform=transform, path_to_dir=path_to_dir)
    print(galaxy10_train.get_stats())
    shuffle_dataset = True
    random_seed = 42
    dataset_size = galaxy10_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(galaxy10_train, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(galaxy10_train, batch_size=batch_size,
                                                    sampler=valid_sampler)

    galaxy10_test = Galaxy10Dataset(mode='test', transform=transform, path_to_dir=path_to_dir)
    test_loader = DataLoader(galaxy10_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader

def get_isic_dataloaders(path_to_dir = "..", validation_split=0.2, batch_size=8):
    if batch_size < 0:
        batch_size = 8
    if not os.path.exists(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_SplitTrain_GroundTruth.csv"):
        if os.path.exists(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv"):
            make_isic_traintest(path_to_dir)
        else:
            print("No data found")
            return None, None, None
    
    train_gt = pd.read_csv(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_SplitTrain_GroundTruth.csv")
    train_gt["target"] = train_gt.apply(lambda x: np.argmax(x[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]]), axis=1)
    #classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    isic_train = ISICDataset(train_gt, path_to_dir, input_size=256)

    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(isic_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(isic_train, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(isic_train, batch_size=batch_size,
                                                    sampler=valid_sampler)

    test_gt = pd.read_csv(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_SplitTest_GroundTruth.csv")
    test_gt["target"] = test_gt.apply(lambda x: np.argmax(x[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]]), axis=1)
    isic_test = ISICDataset(test_gt, path_to_dir, input_size=256)
    test_loader = DataLoader(isic_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader

def make_isic_traintest(path_to_dir = "..", test_split=0.1, seed=42):
    groundtruth_df = pd.read_csv(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv")
    print(groundtruth_df)
    print([str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_Training_Input/{}.jpg".format(im) for im in groundtruth_df["image"]][:10])
    groundtruth_df = groundtruth_df.loc[[os.path.exists(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_Training_Input/{}.jpg".format(im)) for im in groundtruth_df["image"]]]
    groundtruth_df = groundtruth_df[groundtruth_df["image"] != "ISIC_0072651" and groundtruth_df["image"] != "ISIC_0027736"]
    print(groundtruth_df)
    train=groundtruth_df.sample(frac=1-test_split,random_state=seed) 
    test=groundtruth_df.drop(train.index)
    assert len(list(set(train.image) & set(test.image))) == 0
    train.to_csv(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_SplitTrain_GroundTruth.csv", index=False)
    test.to_csv(str(path_to_dir)+"/data/ISIC_2019/ISIC_2019_SplitTest_GroundTruth.csv", index=False)

def make_galaxy10_traintest(path_to_dir = "..", test_split=0.1, seed=42):
    np.random.seed(seed)
    file = str(path_to_dir)+"/data/Galaxy10_DECals.h5"
    f = h5py.File(file, 'r')
    print(f.keys())
    labels, images = f['ans'], f['images']
    inds = np.arange(len(labels))
    np.random.shuffle(inds)
    split_ind = int(np.floor(test_split * len(inds)))
    tv_inds, test_inds = sorted(inds[split_ind:]), sorted(inds[:split_ind])
    #tv_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5", 'w')
    print("tv", tv_inds[0], images[tv_inds,:,:,:].shape, labels[tv_inds].shape)
    # tv_f.create_dataset('images', data=images[tv_inds,:,:,:])
    # tv_f.create_dataset('labels', data=labels[tv_inds])
    # tv_f.close()
    # test_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_test.h5", 'w')
    print("test", test_inds[0], images[test_inds,:,:,:].shape, labels[test_inds].shape)
    # test_f.create_dataset('images', data=images[test_inds,:,:,:])
    # test_f.create_dataset('labels', data=labels[test_inds])
    # test_f.close()

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
