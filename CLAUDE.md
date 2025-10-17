# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Pi-3D-Tracking is a multi-camera 3D tracking system built around the Pi3 model for dense 3D understanding from multiple synchronized camera views. The repository contains the core model implementation, a multi-target multi-camera tracking dataset (MTMC_Tracking_2025), and evaluation framework using HOTA metrics.

## Development Environment

### Python Dependencies
- **Package Manager**: Uses `uv` for Python package management
- **Install Dependencies**: `uv sync`
- **Virtual Environment**: Managed automatically by uv, activated when using uv commands
- **Dependencies**: Defined in `pyproject.toml` (PyTorch, computer vision libraries, ML tools)

### Adding New Dependencies
- Use `uv add <package>` to add new packages
- Dependencies are automatically added to `pyproject.toml`

## Dataset and Evaluation

### Dataset Download
- **Command**: `python data/download_data.py`
- Downloads MTMC_Tracking_2025 dataset from Hugging Face
- Contains multi-camera warehouse, hospital, and lab scenes with 3D object annotations

## Key Architecture Components

### Core Model
- **Main Model**: `pi3/models/pi3.py` - Multi-camera transformer-based architecture
- **Model Layers**: `pi3/models/layers/` - Specialized components (attention, heads, etc.)
- **Utilities**: `pi3/utils/` - Geometry functions and helper utilities

### Dataset Structure
- **Training Data**: `data/MTMC_Tracking_2025/train/` - Warehouse scenes
- **Validation Data**: `data/MTMC_Tracking_2025/val/` - Hospital, lab, and warehouse scenes  
- **Test Data**: `data/MTMC_Tracking_2025/test/` - Warehouse scenes
- **Evaluation Code**: `data/MTMC_Tracking_2025/eval/` - HOTA metrics and TrackEval integration

## Important Development Notes

### Virtual Environment Usage
- Always use `uv` commands which automatically handle virtual environment
- Don't manually activate/deactivate Python virtual environments
- Use `uv run <command>` for running Python scripts

### Dual Environment Setup
- **Main Development**: Use uv environment for model development and training
- **Evaluation Only**: Use conda environment specifically for running evaluation scripts
- Keep these environments separate to avoid dependency conflicts

### Common Commands
- **Install/Update Dependencies**: `uv sync`
- **Add New Package**: `uv add <package-name>`
- **Run Python Script**: `uv run python <script.py>`
- **Download Dataset**: `uv run python data/download_data.py`

### File Organization
- Model implementation in `pi3/` directory
- Dataset and evaluation in `data/` directory
- Dependencies managed through `pyproject.toml`
- Main development uses uv, evaluation uses conda