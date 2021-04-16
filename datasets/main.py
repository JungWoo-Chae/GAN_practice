import os
from os.path import join as ospj
from glob import glob
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image


def get_data_loader(args):
    dataset_name = args.dataset #cfg['dataset_name']
    
    implemented_datasets = ('celeba', 'cifar10')
    
    assert dataset_name in implemented_datasets
    dataset = None
    
    transform = A.Compose(
        [
            A.Resize(height=args.image_size, width=args.image_size, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(p=1.0),
        ])
    
    if dataset_name == 'celeba':
        dataset = CelebADataset(root='../../data/img_align_celeba', train=True, transforms=transform)
    if dataset_name == 'cifar10':
        dataset = Cifar10SearchDataset(root='../../data', train=True,
                                        download=True, transform=transform)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return data_loader


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="../../data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image #, label


class CelebADataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.img_names = glob(ospj(root, '*.jpg'))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx: int):
        img_path = self.img_names[idx]
        image = np.asarray(Image.open(img_path))
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
            
        return image