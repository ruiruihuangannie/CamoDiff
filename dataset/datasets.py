import os
import albumentations as A
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import nibabel as nib
from typing import Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path
import torchvision.transforms as transforms
from glob import glob


class ACTK_dataset(Dataset):
    def __init__(self, image_root, gt_root, size, split='train', mean=None, std=None):
        """
        General dataset class for both training and testing.
        
        Args:
            image_root (str): Path to image directory
            gt_root (str): Path to ground truth directory  
            size (int): Size to resize images (trainsize for train, testsize for test)
            split (str): 'train' or 'test' mode
            mean (list): Normalization mean values
            std (list): Normalization std values
        """
        super().__init__()
        self.resize = size
        self.split = split
        
        self.images = sorted(glob(resolve_path(image_root) / '*.png'))
        self.gts = sorted(glob(resolve_path(gt_root) / '*.png'))
        assert len(self.images) == len(self.gts), f"Mismatch: {len(self.images)} images vs {len(self.gts)} ground truth files"
        
        # Setup transforms
        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])
        
        if self.split == 'train':
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
            ])
        else:
            self.aug_transform = None
            
        self.dataset_size = len(self.images)

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        gt = Image.open(self.gts[idx])
        
        if self.split == 'train' and self.aug_transform:
            image_np = np.array(image)
            gt_np = np.array(gt)
            augmented = self.aug_transform(image=image_np, mask=gt_np)
            image = Image.fromarray(augmented['image'])
            gt = Image.fromarray(augmented['mask'])
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        
        return {'image': image, 'gt': gt}