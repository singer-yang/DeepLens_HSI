import os
import random

# import OpenEXR
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CaveDataset(Dataset):
    """Dataset for the Cave dataset."""

    def __init__(self, file_path, crop_size=(256, 256), is_train=True):
        self.file_path = file_path
        self.is_train = is_train
        self.crop_size = crop_size

        # Load the cavefile list
        with open(file_path, "r") as f:
            self.image_folders = [line.strip() for line in f.readlines()]

        # Try read the first file, if not found, download the dataset
        if not os.path.exists(self.image_folders[0]):
            print(f"File {self.image_folders[0]} not found. Downloading CAVE dataset...")
            from datasets.download_cave import download_cave_dataset
            download_cave_dataset()
            if not os.path.exists(self.image_folders[0]):
                raise FileNotFoundError(f"Could not find or download dataset file: {self.image_folders[0]}")

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
        hsi_tensor = np.stack(images, axis=-1).astype(np.float32)  # [H, W, 31]

        # Data augmentation
        hsi_tensor = self.transform(hsi_tensor)

        # Create data dictionary
        data_dict = {
            "wvln": torch.linspace(0.4, 0.7, 31),
            "img": hsi_tensor,
        }

        return data_dict
