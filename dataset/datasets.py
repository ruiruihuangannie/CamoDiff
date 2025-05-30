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
from dataset.data_val import randomPeper, random_modified
import torch.nn.functional as F

class ACTK_dataset(Dataset):
    def __init__(self, image_root, gt_root, size, split='train', mean=None, std=None, 
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
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        assert len(self.images) == len(self.gts), f"Mismatch: {len(self.images)} images vs {len(self.gts)} ground truth files"

        # Setup transforms
        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])
        
        if self.split == 'train':
            self.aug_transform = A.Compose([
                A.RandomScale(scale_limit=0.25, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.aug_transform = None
            
        self.dataset_size = len(self.images)

    def get_transform(self, mean=None, std=None):
        # For grayscale images, use appropriate normalization
        # Convert to [0,1] range with ToTensor, then normalize to [-1,1] or use ImageNet style
        mean = [0.5] if mean is None else mean  # Normalize [0,1] to [-1,1]
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
        gt = self.binary_loader(self.gts[idx])
        image_size = image.size

        # data augmentation
        if self.split == 'train':
            assert self.aug_transform is not None, "Augmentation is not applied in training mode"
            image_np = np.array(image)
            gt_np = np.array(gt)
            augmented = self.aug_transform(image=image_np, mask=gt_np)
            image, gt = augmented['image'], augmented['mask']
            
            # Pad and crop like in data_val.py
            padded = A.PadIfNeeded(*image_size[::-1], border_mode=0)(image=image, mask=gt)
            cropped = A.RandomCrop(*image_size[::-1])(image=padded['image'], mask=padded['mask'])
            image, gt = cropped['image'], cropped['mask']
            
            # Convert numpy arrays back to PIL Images
            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            
            # Boundary modification (for training)
            if self.do_boundary_modification:
                seg = random_modified(np.array(gt), **self.boundary_args)
                seg = self.gt_transform(Image.fromarray(seg))
                data['seg'] = seg

            gt = randomPeper(np.array(gt)) if self.do_randomPeper else gt
        else:
            image_for_post = image.copy()
            image_for_post = image_for_post.resize(gt.size)
            data['image_for_post'] = self.get_transform()(image_for_post)
            data['name'] = self.images[idx].split('/')[-1]

        data['image'] = self.img_transform(image)
        data['gt'] = self.gt_transform(gt)

        # gt_idx = data['gt'].squeeze(0).long()  
        # # build [H, W, class_num] one-hot, then permute to [class_num, H, W]
        # # currently fixed
        # cn = 4
        # one_hot = F.one_hot(gt_idx, num_classes=cn) \
        #              .permute(2, 0, 1) \
        #              .float()
        # data['gt'] = one_hot

        # if 'seg' in data:
        #     # squeeze off the 1-channel dim
        #     seg_idx = data['seg'].squeeze(0).long()  
        #     # build H×W×4 one-hot, then permute → 4×H×W
        #     seg_oh = F.one_hot(seg_idx, num_classes=cn) \
        #                 .permute(2,0,1).float()
        #     data['seg'] = seg_oh
        return data

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def load_data(self):
        """Load data method for compatibility with test_dataset interface"""

        assert self.split != 'test', "load_data() is only available in test mode"
        image = self.binary_loader(self.images[self.index])
        image_tensor = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        
        image_for_post = image.resize(gt.size)

        self.index += 1
        self.index = self.index % self.dataset_size

        return image_tensor, gt, name, image_for_post

    def __iter__(self):
        """Iterator for compatibility with test_dataset interface"""
        assert self.split != 'test', "Iterator is only available in test mode"
        for i in range(self.dataset_size):
            yield self.load_data()