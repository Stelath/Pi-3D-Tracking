"""
MTMC Tracking Dataset Module

PyTorch dataset and dataloader for Multi-Target Multi-Camera (MTMC) Tracking.

Example usage:
    from modules import create_mtmc_dataloader, MTMCTrackingDataset

    # Quick way: use factory function
    train_loader = create_mtmc_dataloader(
        data_root="data/MTMC_Tracking_2025",
        split="train",
        batch_size=4,
        cache_videos_in_memory=False,
        num_workers=4
    )

    for batch in train_loader:
        images = batch["images"]  # List of lists of [H, W, 3] arrays
        intrinsics = batch["intrinsics"]  # List of [N, 3, 3] arrays
        # ... process batch

    # Manual way: create dataset directly
    dataset = MTMCTrackingDataset(
        data_root="data/MTMC_Tracking_2025",
        split="train",
        cache_videos_in_memory=False
    )

    sample = dataset[0]
    images = sample["images"]  # List of [H, W, 3] arrays (one per camera)
    intrinsics = sample["intrinsics"]  # [N_cameras, 3, 3] array
"""

from .mtmc_dataset import MTMCTrackingDataset, collate_mtmc_batch
from .dataloader import create_mtmc_dataloader, create_train_val_dataloaders
from .video_cache import VideoCache, MultiSceneVideoCache
from .transforms import Pi3Transform

__all__ = [
    # Main dataset class
    "MTMCTrackingDataset",
    # DataLoader factory functions
    "create_mtmc_dataloader",
    "create_train_val_dataloaders",
    # Collate function (for custom DataLoader creation)
    "collate_mtmc_batch",
    # Video cache (for advanced use cases)
    "VideoCache",
    "MultiSceneVideoCache",
    # Transforms
    "Pi3Transform",
]