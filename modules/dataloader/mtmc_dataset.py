"""
Multi-Target Multi-Camera (MTMC) Tracking Dataset.

PyTorch dataset for loading synchronized multi-view frames with calibration
and 3D tracking annotations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset

from .video_cache import MultiSceneVideoCache


class MTMCTrackingDataset(Dataset):
    """
    PyTorch dataset for MTMC Tracking 2025 dataset.

    Returns synchronized frames from all cameras at a given timestamp, along with
    camera calibration and object annotations.

    Example usage:
        dataset = MTMCTrackingDataset(
            data_root="data/MTMC_Tracking_2025",
            split="train",
            cache_videos_in_memory=False
        )

        sample = dataset[0]
        # sample contains:
        # - images: List of [H, W, 3] numpy arrays (one per camera)
        # - intrinsics: [N, 3, 3] array of camera intrinsic matrices
        # - extrinsics: [N, 3, 4] array of camera extrinsic matrices
        # - camera_ids: List of camera ID strings
        # - annotations: List of object annotations for this frame
        # - scene_id: Scene name (e.g., "Warehouse_000")
        # - frame_idx: Frame index in the scene
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        cache_videos_in_memory: bool = False,
        transform: Optional[Callable] = None,
        verbose: bool = True
    ):
        """
        Initialize MTMC Tracking dataset.

        Args:
            data_root: Root directory containing train/val/test folders
            split: Dataset split ("train", "val", or "test")
            cache_videos_in_memory: If True, load all videos into RAM (fast but memory-intensive)
            transform: Optional transform function to apply to images
            verbose: Show progress during initialization
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.cache_videos_in_memory = cache_videos_in_memory
        self.transform = transform
        self.verbose = verbose

        # Get scene paths
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.scene_paths = sorted(split_dir.glob("*/"))

        if len(self.scene_paths) == 0:
            raise ValueError(f"No scenes found in {split_dir}")

        if verbose:
            print(f"Found {len(self.scene_paths)} scenes in {split} split")

        # Load all calibration and ground truth JSONs into memory
        self.calibrations: Dict[str, Dict] = {}
        self.ground_truths: Dict[str, Dict] = {}
        self._load_annotations()

        # Initialize video cache
        self.video_cache = MultiSceneVideoCache(
            scene_paths=self.scene_paths,
            cache_in_memory=cache_videos_in_memory,
            verbose=verbose
        )

        # Build index: list of (scene_id, frame_idx) tuples
        self.index: List[Tuple[str, int]] = []
        self._build_index()

        if verbose:
            print(f"Dataset initialized with {len(self)} frames")

    def _load_annotations(self):
        """Load all calibration and ground truth JSONs into memory."""
        if self.verbose:
            print("Loading annotations...")

        for scene_path in self.scene_paths:
            scene_id = scene_path.name

            # Load calibration
            calib_path = scene_path / "calibration.json"
            if calib_path.exists():
                with open(calib_path, "r") as f:
                    self.calibrations[scene_id] = json.load(f)
            else:
                print(f"Warning: No calibration found for {scene_id}")
                self.calibrations[scene_id] = {"sensors": []}

            # Load ground truth (only available for train split)
            gt_path = scene_path / "ground_truth.json"
            if gt_path.exists():
                with open(gt_path, "r") as f:
                    self.ground_truths[scene_id] = json.load(f)
            else:
                if self.verbose and self.split == "train":
                    print(f"Warning: No ground truth found for {scene_id}")
                self.ground_truths[scene_id] = {}

    def _build_index(self):
        """Build index of all (scene_id, frame_idx) pairs."""
        if self.verbose:
            print("Building frame index...")

        for scene_path in self.scene_paths:
            scene_id = scene_path.name

            # Get number of frames from video cache
            scene_info = self.video_cache.get_scene_info(scene_id)
            if scene_info is None:
                continue

            num_frames = scene_info["num_frames"]

            # Add all frames from this scene to index
            for frame_idx in range(num_frames):
                self.index.append((scene_id, frame_idx))

    def __len__(self) -> int:
        """Return total number of frames across all scenes."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample: all camera views at a specific frame.

        Args:
            idx: Index into the dataset

        Returns:
            Dict with keys:
                - images: List of [H, W, 3] numpy arrays (RGB, one per camera)
                - intrinsics: [N_cameras, 3, 3] numpy array of intrinsic matrices
                - extrinsics: [N_cameras, 3, 4] numpy array of extrinsic matrices
                - camera_ids: List[str] of camera IDs
                - annotations: List of object annotations for this frame
                - scene_id: str, scene name
                - frame_idx: int, frame index in scene
        """
        scene_id, frame_idx = self.index[idx]

        # Get all frames from all cameras at this timestamp
        frames_dict = self.video_cache.get_all_frames_at_timestamp(scene_id, frame_idx)

        # Get calibration data for this scene
        calib = self.calibrations.get(scene_id, {"sensors": []})

        # Extract camera data in sorted order
        camera_ids = sorted(frames_dict.keys())
        images = [frames_dict[cam_id] for cam_id in camera_ids]

        # Apply transforms if provided
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        # Get calibration matrices for each camera
        intrinsics_list = []
        extrinsics_list = []

        for camera_id in camera_ids:
            # Find this camera in calibration data
            sensor_data = self._find_camera_in_calibration(calib, camera_id)

            if sensor_data is not None:
                intrinsics_list.append(np.array(sensor_data["intrinsicMatrix"], dtype=np.float32))
                extrinsics_list.append(np.array(sensor_data["extrinsicMatrix"], dtype=np.float32))
            else:
                # Use identity matrices if calibration not found
                intrinsics_list.append(np.eye(3, dtype=np.float32))
                extrinsics_list.append(np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]))

        intrinsics = np.stack(intrinsics_list, axis=0)  # [N, 3, 3]
        extrinsics = np.stack(extrinsics_list, axis=0)  # [N, 3, 4]

        # Get annotations for this frame
        gt = self.ground_truths.get(scene_id, {})
        annotations = gt.get(str(frame_idx), [])

        return {
            "images": images,  # List of [H, W, 3] arrays
            "intrinsics": intrinsics,  # [N, 3, 3]
            "extrinsics": extrinsics,  # [N, 3, 4]
            "camera_ids": camera_ids,  # List[str]
            "annotations": annotations,  # List[Dict]
            "scene_id": scene_id,  # str
            "frame_idx": frame_idx,  # int
        }

    def _find_camera_in_calibration(self, calib: Dict, camera_id: str) -> Optional[Dict]:
        """
        Find camera calibration data by camera ID.

        The calibration file uses IDs like "Camera_0000" while video files
        might be named "Camera_0000.mp4". We need to match them.

        Args:
            calib: Calibration dictionary
            camera_id: Camera ID to find

        Returns:
            Camera sensor data dict, or None if not found
        """
        # Extract numeric part from camera_id (e.g., "Camera_0000" -> "0000")
        if "Camera_" in camera_id:
            cam_num = camera_id.split("_")[-1]
        else:
            cam_num = camera_id

        # Search for this camera in the sensors list
        for sensor in calib.get("sensors", []):
            if sensor.get("type") != "camera":
                continue

            sensor_id = sensor.get("id", "")
            # Extract numeric part from sensor ID
            if "Camera_" in sensor_id:
                sensor_num = sensor_id.split("_")[-1]
            else:
                sensor_num = sensor_id

            if sensor_num == cam_num or sensor_id == camera_id:
                return sensor

        return None

    def get_scene_ids(self) -> List[str]:
        """Get list of all scene IDs in this dataset."""
        return [p.name for p in self.scene_paths]

    def get_num_frames_in_scene(self, scene_id: str) -> int:
        """Get number of frames in a specific scene."""
        scene_info = self.video_cache.get_scene_info(scene_id)
        if scene_info is None:
            return 0
        return scene_info["num_frames"]

    def get_num_cameras_in_scene(self, scene_id: str) -> int:
        """Get number of cameras in a specific scene."""
        scene_info = self.video_cache.get_scene_info(scene_id)
        if scene_info is None:
            return 0
        return len(scene_info["camera_ids"])


def collate_mtmc_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching MTMC dataset samples.

    Since each sample can have a different number of cameras, we return
    lists instead of stacking into tensors.

    Args:
        batch: List of samples from MTMCTrackingDataset

    Returns:
        Batched dictionary with lists instead of tensors where needed
    """
    if len(batch) == 0:
        return {}

    # Each element is a list (different number of cameras per scene)
    batched = {
        "images": [sample["images"] for sample in batch],
        "intrinsics": [sample["intrinsics"] for sample in batch],
        "extrinsics": [sample["extrinsics"] for sample in batch],
        "camera_ids": [sample["camera_ids"] for sample in batch],
        "annotations": [sample["annotations"] for sample in batch],
        "scene_id": [sample["scene_id"] for sample in batch],
        "frame_idx": [sample["frame_idx"] for sample in batch],
    }

    return batched