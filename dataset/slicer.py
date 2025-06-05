import os
import sys

import numpy as np
from PIL import Image
import torch
import nibabel as nib
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path

class Slicer:
    """
    Slice 3D medical images (e.g., .nii.gz files) into 2D slices and organize them into train/test/validation sets.
      
    Args:
        image_root (str): Path to directory containing 3D image files
        gt_root (str): Path to directory containing 3D ground truth/mask files  
        split (list): Train/test/validation split ratios as [train, test, val] summing to 10
        num_classes (int): Number of classes in the ground truth
        format (str): File extension to process (default: '.nii.gz')
        verbose (bool): Whether to print progress information
        
    Raises:
        ValueError: If split ratios don't sum to 10 or wrong number of splits provided
        AssertionError: If number of image and ground truth files don't match
    """
    def __init__(self, image_root, gt_root, split = [5, 3, 2], num_classes = 5, format = '.nii.gz', verbose = True):
      self.image_root = resolve_path(image_root)
      self.gt_root = resolve_path(gt_root)
      self.format = format
      self.num_classes = num_classes
      self.verbose = verbose
      self.image_slice_dir = os.path.join(self.image_root, 'slices')
      self.gt_slice_dir = os.path.join(self.gt_root, 'slices')

      if self.verbose:
          print(f"[Info] Loading images from: {self.image_root}")
          print(f"[Info] Loading ground truth from: {self.gt_root}")
          print(f"[Info] Output image slices to: {self.image_slice_dir}")
          print(f"[Info] Output ground truth to: {self.gt_slice_dir}")

      self.raw_3d_images = [os.path.join(self.image_root, f) for f in os.listdir(self.image_root) if f.endswith(self.format)]
      self.raw_3d_gts = [os.path.join(self.gt_root, f) for f in os.listdir(self.gt_root) if f.endswith(self.format)]
      self.raw_3d_images.sort()
      self.raw_3d_gts.sort()

      assert len(self.raw_3d_images) == len(self.raw_3d_gts), f"Mismatch: {len(self.raw_3d_images)} images vs {len(self.raw_3d_gts)} ground truth files"
      size_3d = len(self.raw_3d_images)
      
      if len(split) != 3 or sum(split) != 10:
          raise ValueError("Split must be a list of 3 numbers that sum to 10 (e.g., [5, 3, 2])")

      train = int(size_3d / 10.0 * split[0])
      test  = int(size_3d / 10.0 * split[1])
      val   = size_3d - train - test
      self.split = [train, test, val]

      if len(self.raw_3d_images) > 0:
        if self.verbose:
            print(f"[Info] Extracting slices from image {self.format} files...")
        self._extract_images()
        if self.verbose:
            print(f"[Info] Extracting slices from ground truth {self.format} files...")
        self._extract_gts()

    def _extract_images(self):
        """ Extract 2D slices from nii.gz files and save as PNG images. """

        self.tr_folder = os.path.join(self.image_slice_dir, "train")
        self.ts_folder = os.path.join(self.image_slice_dir, "test")
        self.vl_folder = os.path.join(self.image_slice_dir, "val")

        os.makedirs(self.tr_folder, exist_ok=True)
        os.makedirs(self.ts_folder, exist_ok=True)
        os.makedirs(self.vl_folder, exist_ok=True)

        for idx, fn in tqdm(enumerate(self.raw_3d_images), desc='Slicing images', total=len(self.raw_3d_images)):
            img_nib = nib.load(fn)
            img_data = img_nib.get_fdata()
            
            if idx < self.split[0]:
                target_folder = self.tr_folder
            elif idx < self.split[0] + self.split[1]:
                target_folder = self.ts_folder
            else:
                target_folder = self.vl_folder
            
            p_low, p_high = np.percentile(img_data, (1, 99.5))
            for i in range(img_data.shape[2]):
                img_slice = img_data[:, :, i]
                base_name = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]  # Remove .nii.gz
                img_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.png")
                
                img_slice = np.clip(img_slice, p_low, p_high)
                img_slice = ((img_slice - p_low) / (p_high - p_low)) * 255.0
                img_slice = img_slice.astype(np.uint8)
                Image.fromarray(img_slice, mode="L").save(img_slice_path)

    def _extract_gts(self):
        """ Extract 2D slices from nii.gz files and save as PNG images. """

        self.tr_folder = os.path.join(self.gt_slice_dir, "train")
        self.ts_folder = os.path.join(self.gt_slice_dir, "test")
        self.vl_folder = os.path.join(self.gt_slice_dir, "val")

        os.makedirs(self.tr_folder, exist_ok=True)
        os.makedirs(self.ts_folder, exist_ok=True)
        os.makedirs(self.vl_folder, exist_ok=True)

        for idx, fn in tqdm(enumerate(self.raw_3d_gts), desc='Slicing ground truth', total=len(self.raw_3d_gts)):
            gt_nib = nib.load(fn)
            gt_data = gt_nib.get_fdata()
            
            if idx < self.split[0]:
                target_folder = self.tr_folder
            elif idx < self.split[0] + self.split[1]:
                target_folder = self.ts_folder
            else:
                target_folder = self.vl_folder

            for i in range(gt_data.shape[2]):
                gt_slice = gt_data[:, :, i]
                base_name = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]  # Remove .nii.gz
                gt_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.pt")
                
                gt_slice = gt_slice.astype(np.uint8)
                one_hot = torch.zeros((self.num_classes, gt_slice.shape[0], gt_slice.shape[1]))
                for i in range(self.num_classes):
                    one_hot[i][gt_slice == i] = 1
                
                torch.save(one_hot, gt_slice_path)

def test_gt_slicer(fn):
    gt_file = torch.load(fn)
    composite_array = np.zeros((gt_file.shape[1], gt_file.shape[2]), dtype=np.uint8)
    for i in range(gt_file.shape[0]):
        channel_data = gt_file[i].numpy()
        composite_array[channel_data == 1] = i * 50
    composite_img = Image.fromarray(composite_array.astype(np.uint8))
    output_path = fn.replace('.pt', '.png')
    composite_img.save(output_path)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # python slicer.py /path/to/ground_truth_file.pt
        test_gt_slicer(sys.argv[1])
    elif len(sys.argv) == 3:
        # python slicer.py /path/to/images /path/to/ground_truth
        ACTK_slicer = Slicer(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python slicer.py <image_root> <gt_root>")