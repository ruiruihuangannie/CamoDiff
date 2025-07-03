import os
import sys
from glob import glob
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path
from dataset.data_val import randomPeper, random_modified

class ACTK_dataset(Dataset):
    def __init__(self, image_root, gt_root, size, split, mean=None, std=None, 
                 randomPeper=True, boundary_modification=False, boundary_args={}):
        """
        General dataset class for both training and testing.
        
        Args:
            image_root (str): Path to image directory
            gt_root (str): Path to ground truth directory  
            size (int): Size to resize images (trainsize for train, testsize for test)
            split (str): 'train' or 'test' mode
            mean (list): Normalization mean values
            std (list): Normalization std values
            randomPeper (bool): Whether to apply random pepper noise
            boundary_modification (bool): Whether to apply boundary modification
            boundary_args (dict): Arguments for boundary modification
        """
        super().__init__()
        self.resize = size
        self.split = split
        self.index = 0
        self.do_randomPeper = randomPeper
        self.do_boundary_modification = boundary_modification
        self.boundary_args = boundary_args
        
        image_root = resolve_path(image_root)
        gt_root = resolve_path(gt_root)
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.pt')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        assert len(self.images) == len(self.gts), f"Mismatch: {len(self.images)} images vs {len(self.gts)} ground truth files"

        # Setup transforms
        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        
        if self.split == 'train':
            self.aug_transform = transforms.Compose([
                transforms.RandomAffine(degrees=15, scale=(0.75, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90)
            ])
        else:
            self.aug_transform = None
            
        self.dataset_size = len(self.images)

    def get_transform(self, mean=None, std=None):
        mean = [0.5] if mean is None else mean
        std = [0.5] if std is None else std
        
        transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        data = {}
        image = self.binary_loader(self.images[idx])
        gt = torch.load(self.gts[idx])
        
        image = transforms.Resize((self.resize, self.resize))(image)
        gt = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(gt)

        # data augmentation
        if self.split == 'train':
            # Use the same random seed for both image and gt transformations
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            gt = self.aug_transform(torch.unsqueeze(gt, 0))
            gt = torch.squeeze(gt, 0)
            
            # Boundary modification (for training)
            if self.do_boundary_modification:
                data['seg'] = random_modified(gt, **self.boundary_args)
            
            if self.do_randomPeper:
                gt_np = gt.numpy()
                for c in range(gt_np.shape[0]):
                    gt_np[c] = randomPeper(gt_np[c])
                gt = torch.from_numpy(gt_np)
        else:
            image_for_post = image.copy()
            data['image_for_post'] = self.get_transform()(image_for_post)
            data['name'] = self.images[idx].split('/')[-1]

        data['image'] = self.img_transform(image)
        data['gt'] = gt

        return data

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def load_data(self):
        """Load data method for compatibility with test_dataset interface"""

        assert self.split == 'test', "load_data() is only available in test mode"
        image = self.binary_loader(self.images[self.index])
        image_tensor = self.img_transform(image).unsqueeze(0)
        gt = torch.load(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        
        image_for_post = image.resize(gt.size)

        self.index += 1
        self.index = self.index % self.dataset_size

        return image_tensor, gt, name, image_for_post

    def __iter__(self):
        """Iterator for compatibility with test_dataset interface"""
        assert self.split == 'test', "Iterator is only available in test mode"
        for i in range(self.dataset_size):
            yield self.load_data()