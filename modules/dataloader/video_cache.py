"""
Video caching utilities for efficient multi-view video loading.

Supports two modes:
1. Direct I/O: Read frames on-demand from disk (low memory, slower)
2. Shared Memory: Pre-load all videos into RAM (high memory, fast)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm


class VideoCache:
    """
    Video caching system that can load videos on-demand or cache them in memory.

    Supports two modes:
    - cache_in_memory=False: Read frames directly from MP4 files
    - cache_in_memory=True: Pre-load all frames into numpy arrays in RAM
    """

    def __init__(
        self,
        scene_path: Path,
        cache_in_memory: bool = False,
        verbose: bool = False
    ):
        """
        Initialize video cache for a single scene.

        Args:
            scene_path: Path to scene directory containing videos/ folder
            cache_in_memory: If True, load all videos into memory
            verbose: Show progress bars during loading
        """
        self.scene_path = Path(scene_path)
        self.video_dir = self.scene_path / "videos"
        self.cache_in_memory = cache_in_memory
        self.verbose = verbose

        # Find all video files
        self.video_paths = sorted(self.video_dir.glob("*.mp4"))
        self.camera_ids = [p.stem for p in self.video_paths]

        # Memory cache (only used if cache_in_memory=True)
        self._cached_videos: Dict[str, np.ndarray] = {}

        # Video capture objects (only used if cache_in_memory=False)
        self._video_captures: Dict[str, cv2.VideoCapture] = {}

        # Get video metadata
        if len(self.video_paths) > 0:
            cap = cv2.VideoCapture(str(self.video_paths[0]))
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            self.num_frames = 0
            self.frame_width = 0
            self.frame_height = 0
            self.fps = 0

        # Load videos into memory if requested
        if self.cache_in_memory:
            self._load_all_videos()

    def _load_all_videos(self):
        """Load all videos into memory as numpy arrays."""
        desc = f"Loading videos for {self.scene_path.name}" if self.verbose else None

        for video_path in tqdm(self.video_paths, desc=desc, disable=not self.verbose):
            camera_id = video_path.stem
            frames = self._read_all_frames(video_path)
            self._cached_videos[camera_id] = frames

    def _read_all_frames(self, video_path: Path) -> np.ndarray:
        """Read all frames from a video file into a numpy array."""
        cap = cv2.VideoCapture(str(video_path))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Stack into [T, H, W, C] array
        frames = np.stack(frames, axis=0)
        return frames

    def get_frame(self, camera_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from a specific camera.

        Args:
            camera_id: Camera ID (e.g., "Camera_0000")
            frame_idx: Frame index

        Returns:
            Frame as numpy array [H, W, 3] in RGB format, or None if not found
        """
        if camera_id not in self.camera_ids:
            return None

        if self.cache_in_memory:
            # Get from memory cache
            if camera_id in self._cached_videos:
                if 0 <= frame_idx < len(self._cached_videos[camera_id]):
                    return self._cached_videos[camera_id][frame_idx].copy()
            return None
        else:
            # Read from disk on-demand
            return self._read_frame_from_disk(camera_id, frame_idx)

    def _read_frame_from_disk(self, camera_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame directly from video file."""
        # Get or create video capture object
        if camera_id not in self._video_captures:
            video_path = self.video_dir / f"{camera_id}.mp4"
            if not video_path.exists():
                return None
            self._video_captures[camera_id] = cv2.VideoCapture(str(video_path))

        cap = self._video_captures[camera_id]

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_all_frames_at_timestamp(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """
        Get frames from all cameras at a specific timestamp.

        Args:
            frame_idx: Frame index

        Returns:
            Dict mapping camera_id -> frame array [H, W, 3]
        """
        frames = {}
        for camera_id in self.camera_ids:
            frame = self.get_frame(camera_id, frame_idx)
            if frame is not None:
                frames[camera_id] = frame
        return frames

    def __del__(self):
        """Cleanup: release all video capture objects."""
        for cap in self._video_captures.values():
            if cap is not None:
                cap.release()
        self._video_captures.clear()


class MultiSceneVideoCache:
    """
    Video cache manager for multiple scenes.

    Efficiently manages video loading across multiple scenes with optional
    memory caching.
    """

    def __init__(
        self,
        scene_paths: List[Path],
        cache_in_memory: bool = False,
        verbose: bool = True
    ):
        """
        Initialize multi-scene video cache.

        Args:
            scene_paths: List of paths to scene directories
            cache_in_memory: If True, load all videos into memory
            verbose: Show progress bars during loading
        """
        self.scene_paths = [Path(p) for p in scene_paths]
        self.cache_in_memory = cache_in_memory
        self.verbose = verbose

        # Create a VideoCache for each scene
        self.scene_caches: Dict[str, VideoCache] = {}

        if verbose:
            print(f"Initializing video cache for {len(scene_paths)} scenes...")
            if cache_in_memory:
                print("Loading all videos into memory (this may take a while)...")

        for scene_path in tqdm(scene_paths, desc="Loading scenes", disable=not verbose):
            scene_id = scene_path.name
            self.scene_caches[scene_id] = VideoCache(
                scene_path=scene_path,
                cache_in_memory=cache_in_memory,
                verbose=False  # Disable per-scene progress bars for cleaner output
            )

        if verbose and cache_in_memory:
            total_gb = sum(
                sum(v.nbytes for v in cache._cached_videos.values())
                for cache in self.scene_caches.values()
            ) / (1024 ** 3)
            print(f"Total video cache size: {total_gb:.2f} GB")

    def get_frame(self, scene_id: str, camera_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from a specific scene and camera.

        Args:
            scene_id: Scene ID (e.g., "Warehouse_000")
            camera_id: Camera ID (e.g., "Camera_0000")
            frame_idx: Frame index

        Returns:
            Frame as numpy array [H, W, 3] in RGB format, or None if not found
        """
        if scene_id not in self.scene_caches:
            return None
        return self.scene_caches[scene_id].get_frame(camera_id, frame_idx)

    def get_all_frames_at_timestamp(
        self,
        scene_id: str,
        frame_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Get frames from all cameras in a scene at a specific timestamp.

        Args:
            scene_id: Scene ID (e.g., "Warehouse_000")
            frame_idx: Frame index

        Returns:
            Dict mapping camera_id -> frame array [H, W, 3]
        """
        if scene_id not in self.scene_caches:
            return {}
        return self.scene_caches[scene_id].get_all_frames_at_timestamp(frame_idx)

    def get_scene_info(self, scene_id: str) -> Optional[Dict]:
        """
        Get metadata about a scene.

        Returns:
            Dict with keys: num_frames, frame_width, frame_height, fps, camera_ids
        """
        if scene_id not in self.scene_caches:
            return None

        cache = self.scene_caches[scene_id]
        return {
            "num_frames": cache.num_frames,
            "frame_width": cache.frame_width,
            "frame_height": cache.frame_height,
            "fps": cache.fps,
            "camera_ids": cache.camera_ids,
        }