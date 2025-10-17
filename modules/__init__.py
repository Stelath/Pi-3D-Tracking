"""Modules for 3D detection training and inference.

This package contains modular components organized by functionality:
- constants: Box format and quality metric constants
- ops: Neural network operations
- initialization: Weight initialization utilities
- loss: Loss functions
- target: Target assignment modules (includes base classes)
- dataloader: Data loading utilities
"""

# Import from submodules
from modules.constants import (
    X, Y, Z, W, L, H,
    SIN_YAW, COS_YAW, YAW,
    VX, VY, VZ,
    CNS, YNS,
)
from modules.ops import Scale, linear_relu_ln
from modules.initialization import bias_init_with_prob, xavier_init
from modules.loss import SparseBox3DLoss
from modules.target import BaseTargetWithDenoising, SparseBox3DTarget

__all__ = [
    # Constants
    'X', 'Y', 'Z', 'W', 'L', 'H',
    'SIN_YAW', 'COS_YAW', 'YAW',
    'VX', 'VY', 'VZ',
    'CNS', 'YNS',
    # Operations
    'Scale',
    'linear_relu_ln',
    # Initialization
    'bias_init_with_prob',
    'xavier_init',
    # Loss and Target
    'SparseBox3DLoss',
    'BaseTargetWithDenoising',
    'SparseBox3DTarget',
]
