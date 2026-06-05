"""
dataset.py — DeepPermNet Dataset and DataModule

Loads Mini-ImageNet from HuggingFace, splits each image into a 3x3 grid of 64x64
patches, shuffles them randomly, and returns (shuffled_patches, permutation_target).
"""

import os
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


class DeepPermNetDataset(Dataset):
    """
    Wraps a HuggingFace image dataset.
    For each image:
      1. Resize to 192x192
      2. Split into 9 patches of 64x64
      3. Shuffle patches with a random permutation
    Returns (shuffled_patches, target_permutation) where:
      - shuffled_patches: Tensor of shape (9, 3, 64, 64)
      - target_permutation: LongTensor of shape (9,)
    """

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.base_transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_tensor = self.base_transform(img)  # (3, 192, 192)

        # Extract 3x3 grid → (Channels, Grid_H, Grid_W, Patch_H, Patch_W)
        patches = img_tensor.unfold(1, 64, 64).unfold(2, 64, 64)
        # Reshape to (9, 3, 64, 64)
        patches = patches.contiguous().view(3, -1, 64, 64).permute(1, 0, 2, 3)

        permutation = list(range(9))
        random.shuffle(permutation)
        shuffled_patches = patches[permutation]
        target_permutation = torch.tensor(permutation, dtype=torch.long)

        return shuffled_patches, target_permutation


class DeepPermNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DeepPermNet.
    Reads dataset name, batch size, and num_workers from environment variables.
    """

    def __init__(
        self,
        dataset_name: str = None,
        batch_size: int = None,
        num_workers: int = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name or os.getenv("DATASET_NAME", "timm/mini-imagenet")
        self.batch_size = batch_size or int(os.getenv("BATCH_SIZE", 32))
        self.num_workers = num_workers or int(os.getenv("NUM_WORKERS", 2))

    def setup(self, stage=None):
        hf_token = os.getenv("HF_TOKEN")
        ds = load_dataset(self.dataset_name, token=hf_token)
        self.train_dataset = DeepPermNetDataset(ds["train"])
        self.val_dataset = DeepPermNetDataset(ds["validation"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
