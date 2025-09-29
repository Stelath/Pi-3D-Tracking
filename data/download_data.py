#!/usr/bin/env python3
"""
Download script for MTMC_Tracking_2025 dataset from Hugging Face.
Downloads all data except depth_maps folders to the data/ directory.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm
import time


class DatasetDownloader:
    def __init__(self, repo_id: str, local_dir: str = "data"):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.api = HfApi()
        self.base_path = "MTMC_Tracking_2025"
        self.target_folders = ["test", "train", "eval", "val"]
        self.exclude_folder = "depth_maps"
        
    def get_filtered_files(self) -> List[str]:
        """Get list of files to download, excluding depth_maps folders."""
        print("Listing repository files...")
        try:
            all_files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Error listing repository files: {e}")
            return []
        
        # Filter files based on criteria
        filtered_files = []
        for file_path in all_files:
            # Must be in MTMC_Tracking_2025 directory
            if not file_path.startswith(f"{self.base_path}/"):
                continue
                
            # Must be in one of the target folders
            path_parts = file_path.split("/")
            if len(path_parts) < 3:  # Need at least MTMC_Tracking_2025/folder/file
                continue
                
            folder = path_parts[1]
            if folder not in self.target_folders:
                continue
                
            # Exclude depth_maps folders
            if self.exclude_folder in path_parts:
                continue
                
            filtered_files.append(file_path)
        
        return filtered_files
    
    def file_exists_and_complete(self, local_path: Path, expected_size: Optional[int] = None) -> bool:
        """Check if file exists locally and is complete."""
        if not local_path.exists():
            return False
            
        if expected_size is not None:
            return local_path.stat().st_size == expected_size
            
        return True
    
    def download_file(self, file_path: str, max_retries: int = 3) -> bool:
        """Download a single file with retry logic."""
        local_path = self.local_dir / file_path
        
        # Skip if file already exists
        if self.file_exists_and_complete(local_path):
            return True
            
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                print(f"Downloading {file_path} (attempt {attempt + 1}/{max_retries})")
                
                # Download to temporary file first
                temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
                
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(self.local_dir),
                    local_dir_use_symlinks=False
                )
                
                # Move from the hub cache to our desired location
                if Path(downloaded_path) != local_path:
                    if local_path.exists():
                        local_path.unlink()
                    Path(downloaded_path).rename(local_path)
                
                return True
                
            except Exception as e:
                print(f"Error downloading {file_path} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to download {file_path} after {max_retries} attempts")
                    return False
        
        return False
    
    def download_all(self):
        """Download all filtered files."""
        # Get list of files to download
        files_to_download = self.get_filtered_files()
        
        if not files_to_download:
            print("No files found to download!")
            return
        
        print(f"Found {len(files_to_download)} files to download")
        print(f"Excluding files in '{self.exclude_folder}' folders")
        print(f"Download location: {self.local_dir.absolute()}")
        
        # Create base directory
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        successful = 0
        failed = 0
        
        with tqdm(files_to_download, desc="Downloading files") as pbar:
            for file_path in pbar:
                pbar.set_description(f"Downloading {Path(file_path).name}")
                
                if self.download_file(file_path):
                    successful += 1
                else:
                    failed += 1
                
                pbar.set_postfix({
                    "Success": successful,
                    "Failed": failed
                })
        
        print(f"\nDownload complete!")
        print(f"Successfully downloaded: {successful} files")
        print(f"Failed downloads: {failed} files")


def main():
    # Configuration
    repo_id = "nvidia/PhysicalAI-SmartSpaces"
    local_dir = "./"
    
    # Create downloader and start download
    downloader = DatasetDownloader(repo_id, local_dir)
    
    try:
        downloader.download_all()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()