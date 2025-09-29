"""
DataLoader factory functions for MTMC Tracking dataset.

Provides convenient functions to create properly configured DataLoaders
with appropriate settings for multi-worker loading and batching.
"""

from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader

from .mtmc_dataset import MTMCTrackingDataset, collate_mtmc_batch


def create_mtmc_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 1,
    cache_videos_in_memory: bool = False,
    transform: Optional[Callable] = None,
    num_workers: int = 0,
    shuffle: bool = None,
    pin_memory: bool = True,
    max_scenes: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for MTMC Tracking dataset.

    Args:
        data_root: Root directory containing train/val/test folders
        split: Dataset split ("train", "val", or "test")
        batch_size: Batch size (default: 1)
        cache_videos_in_memory: If True, load all videos into RAM
        transform: Optional transform function to apply to images
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle data (default: True for train, False otherwise)
        pin_memory: Whether to pin memory for faster GPU transfer
        max_scenes: If specified, only load first N scenes
        verbose: Show progress during initialization
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Configured DataLoader instance

    Example:
        # Training dataloader with video caching
        train_loader = create_mtmc_dataloader(
            data_root="data/MTMC_Tracking_2025",
            split="train",
            batch_size=4,
            cache_videos_in_memory=True,
            num_workers=4,
            shuffle=True
        )

        # Validation dataloader without caching (to save memory)
        val_loader = create_mtmc_dataloader(
            data_root="data/MTMC_Tracking_2025",
            split="val",
            batch_size=1,
            cache_videos_in_memory=False,
            num_workers=2,
            shuffle=False
        )
    """
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == "train")

    # Create dataset
    dataset = MTMCTrackingDataset(
        data_root=data_root,
        split=split,
        cache_videos_in_memory=cache_videos_in_memory,
        transform=transform,
        max_scenes=max_scenes,
        verbose=verbose
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mtmc_batch,
        **kwargs
    )

    return dataloader


def create_train_val_dataloaders(
    data_root: str,
    train_batch_size: int = 4,
    val_batch_size: int = 1,
    cache_videos_in_memory: bool = False,
    transform: Optional[Callable] = None,
    num_workers: int = 4,
    max_scenes: Optional[int] = None,
    verbose: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Create both training and validation dataloaders.

    Args:
        data_root: Root directory containing train/val/test folders
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        cache_videos_in_memory: If True, load all videos into RAM
        transform: Optional transform function to apply to images
        num_workers: Number of worker processes for data loading
        max_scenes: If specified, only load first N scenes per split
        verbose: Show progress during initialization

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        train_loader, val_loader = create_train_val_dataloaders(
            data_root="data/MTMC_Tracking_2025",
            train_batch_size=8,
            val_batch_size=1,
            cache_videos_in_memory=False,
            num_workers=4
        )
    """
    train_loader = create_mtmc_dataloader(
        data_root=data_root,
        split="train",
        batch_size=train_batch_size,
        cache_videos_in_memory=cache_videos_in_memory,
        transform=transform,
        num_workers=num_workers,
        shuffle=True,
        max_scenes=max_scenes,
        verbose=verbose
    )

    val_loader = create_mtmc_dataloader(
        data_root=data_root,
        split="val",
        batch_size=val_batch_size,
        cache_videos_in_memory=cache_videos_in_memory,
        transform=transform,
        num_workers=num_workers,
        shuffle=False,
        max_scenes=max_scenes,
        verbose=verbose
    )

    return train_loader, val_loader