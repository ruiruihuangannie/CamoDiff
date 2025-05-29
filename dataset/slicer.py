import os, sys
from PIL import Image
import numpy as np
import nibabel as nib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.init_env import resolve_path
from tqdm import tqdm

def nii_slicer(image_root: str, gt_root: str, split = [5, 3, 2], format = '.nii.gz', verbose = True):
    """
    Slice 3D medical images (e.g., .nii.gz files) into 2D slices and organize them into train/test/validation sets.
    
    This function takes 3D medical imaging data and ground truth masks, extracts 2D slices from each volume,
    and saves them as PNG images organized into training, testing, and validation directories.
    
    Args:
        image_root (str): Path to directory containing 3D image files
        gt_root (str): Path to directory containing 3D ground truth/mask files  
        split (list): Train/test/validation split ratios as [train, test, val] summing to 10
        format (str): File extension to process (default: '.nii.gz')
        verbose (bool): Whether to print progress information
        
    Raises:
        ValueError: If split ratios don't sum to 10 or wrong number of splits provided
        AssertionError: If number of image and ground truth files don't match
    """
    image_root = resolve_path(image_root)
    gt_root    = resolve_path(gt_root)
    image_slice_dir = os.path.join(image_root, 'slices')
    gt_slice_dir    = os.path.join(gt_root, 'slices')

    if verbose:
        print(f"[Info] Loading images from: {image_root}")
        print(f"[Info] Loading ground truth from: {gt_root}")
        print(f"[Info] Output image slices to: {image_slice_dir}")
        print(f"[Info] Output ground truth to: {gt_slice_dir}")

    # Get all files from input directories and sort for consistent pairing
    raw_3d_images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(format)]
    raw_3d_gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(format)]
    raw_3d_images.sort()
    raw_3d_gts.sort()

    # Ensure we have matching number of images and ground truth files
    assert len(raw_3d_images) == len(raw_3d_gts), f"Mismatch: {len(raw_3d_images)} images vs {len(raw_3d_gts)} ground truth files"
    size_3d = len(raw_3d_images)
    
    # Validate split ratios
    if len(split) != 3 or sum(split) != 10:
        raise ValueError("Split must be a list of 3 numbers that sum to 10 (e.g., [5, 3, 2])")

    # Calculate actual file counts for each split
    train = int(size_3d / 10.0 * split[0])
    test  = int(size_3d / 10.0 * split[1])
    val   = size_3d - train - test  # Ensure all files are included
    split_counts = [train, test, val]

    if len(raw_3d_images) > 0:
        if verbose:
            print(f"[Info] Extracting slices from image {format} files...")
        _extract_and_store_slices(raw_3d_images, image_slice_dir, split_counts)
        if verbose:
            print(f"[Info] Extracting slices from ground truth {format} files...")
        _extract_and_store_slices(raw_3d_gts, gt_slice_dir, split_counts)

def _extract_and_store_slices(input_files, output_dir, split_counts):
    """
    Extract 2D slices from 3D nii.gz files and save as PNG images.
    
    Args:
        input_files (list): List of paths to 3D NIfTI files
        output_dir (str): Base directory to save sliced images
        split_counts (list): [train_count, test_count, val_count] for file assignment
    """
    tr_folder = os.path.join(output_dir, "train")
    ts_folder = os.path.join(output_dir, "test")
    vl_folder = os.path.join(output_dir, "val")

    os.makedirs(tr_folder, exist_ok=True)
    os.makedirs(ts_folder, exist_ok=True)
    os.makedirs(vl_folder, exist_ok=True)

    for idx, fn in tqdm(enumerate(input_files), desc='Slicing', total=len(input_files)):
        img_nib = nib.load(fn)
        img_data = img_nib.get_fdata()
        
        if idx < split_counts[0]:  # Train
            target_folder = tr_folder
        elif idx < split_counts[0] + split_counts[1]:  # Test
            target_folder = ts_folder
        else:  # Validation
            target_folder = vl_folder
    
        for i in range(img_data.shape[2]):
            img_slice = img_data[:, :, i]
            base_name = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]  # Remove .nii.gz
            img_slice_path = os.path.join(target_folder, f"{base_name}_slice_{i:03d}.png")
            
            img_min = np.min(img_slice)
            img_max = np.max(img_slice)
            img_slice = ((img_slice - img_min) * 255 / (img_max - img_min)).astype(np.uint8) if img_max > img_min else np.zeros_like(img_slice, dtype=np.uint8)
            Image.fromarray(img_slice, mode = "L").save(img_slice_path)

if __name__ == '__main__':
    # python slicer.py /path/to/images /path/to/ground_truth
    if len(sys.argv) >= 3:
        nii_slicer(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python slicer.py <image_root> <gt_root>")