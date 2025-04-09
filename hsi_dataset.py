import torch
from torch.utils.data import Dataset
import os
import imageio
import numpy as np
import random
from torchvision import transforms

# import OpenEXR
import cv2


class KaistDataset(Dataset):
    """Dataset for the KAIST Hyperspectral dataset."""

    def __init__(self, file_path, crop_size=(256, 256), is_train=True):
        """
        Args:
            txt_file (str): Path to the text file listing absolute image paths.
            crop_size (tuple): Desired output resolution (height, width) after cropping.
            is_train (bool): If True, applies training augmentations.
        """
        self.crop_size = crop_size
        self.is_train = is_train

        # Read image paths from txt_file
        with open(file_path, "r") as f:
            self.img_paths = [line.strip() for line in f if line.strip()]

        # Transformations
        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(
                        degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1)
                    ),
                    transforms.RandomResizedCrop(
                        size=self.crop_size, scale=(0.75, 1.0)
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.img_paths)

    def __getitem__(self, index):
        """Fetches the sample at the given index and applies transforms.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            torch.Tensor: The transformed hyperspectral image tensor (C, H, W).
        """
        img_path = self.img_paths[index]

        # Load EXR image
        try:
            # img_hsi = imageio.imread(img_path, format="EXR-FI")
            # img_hsi = cv2.imread(img_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # img_hsi = np.array(img_hsi, dtype=np.float32)
            exr_file = OpenEXR.InputFile(img_path)
            header = exr_file.header()
            dw = header["dataWindow"]
            size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            channels = header["channels"]
            channel_names = sorted(channels.keys())
            print(f"Image dimensions (height, width): {size}")
            print(f"Found {len(channel_names)} channels: {', '.join(channel_names)}")

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise IOError(f"Could not load image: {img_path}") from e

        breakpoint()

        # Handle potential NaN/Inf values and clip
        img_hsi = np.nan_to_num(img_hsi, nan=0.0, posinf=1.0, neginf=0.0)
        img_hsi = np.clip(img_hsi, 0.0, None)

        # Transpose from (H, W, C) to (C, H, W)
        img_hsi = np.transpose(img_hsi, (2, 0, 1))

        # Apply the specified transformation
        if self.transform:
            img_tensor = self.transform(img_hsi, self.crop_size)
        else:
            # Default: just convert to tensor if no transform provided
            img_tensor = torch.from_numpy(np.ascontiguousarray(img_hsi))

        return img_tensor


class CaveDataset(Dataset):
    """Dataset for the Cave dataset."""

    def __init__(self, file_path, crop_size=(256, 256), is_train=True):
        """
        Args:
            root_dir (str): Root directory containing the dataset
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.file_path = file_path
        self.is_train = is_train
        self.crop_size = crop_size

        # Load the appropriate file list
        with open(file_path, "r") as f:
            self.image_folders = [line.strip() for line in f.readlines()]

        # Transformations
        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(
                        degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1)
                    ),
                    transforms.RandomResizedCrop(
                        size=self.crop_size, scale=(0.75, 1.0)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(self.crop_size),
                ]
            )

    def __len__(self):
        """Return the total number of samples"""
        return len(self.image_folders)

    def __getitem__(self, idx):
        """
        Get the hyperspectral image stack for the given index

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            torch.Tensor: Hyperspectral image stack of shape [31, H, W]
        """
        folder_path = self.image_folders[idx]

        # Get all image files in the folder and sort them
        image_files = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith((".png"))
            ]
        )

        # Read all 31 channels
        images = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            images.append(img)
        hsi_tensor = np.stack(images, axis=-1).astype(np.float32) # [H, W, 31]

        # Data augmentation
        hsi_tensor = self.transform(hsi_tensor)

        # Create data dictionary
        data_dict = {
            "wvln": torch.linspace(0.4, 0.7, 31),
            "img": hsi_tensor,
        }
        
        return data_dict
