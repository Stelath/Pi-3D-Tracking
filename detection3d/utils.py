"""Utilities for 3D detection module - plain PyTorch implementation."""
import torch
import torch.nn as nn
import math


# 3D Box representation constants
# Box format: [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]
X = 0
Y = 1
Z = 2
W = 3
L = 4
H = 5
SIN_YAW = 6
COS_YAW = 7
VX = 8
VY = 9
VZ = 10

# Alternative representation
YAW = 6  # Used when yaw is stored as angle instead of sin/cos

# Quality indices
CNS = 0  # Centerness
YNS = 1  # Yawness


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable parameter.
    Replaces mmcv.cnn.Scale.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        if isinstance(scale, (list, tuple)):
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def linear_relu_ln(embed_dims, in_loops=1, out_loops=1, input_dims=None):
    """Create a sequence of Linear-ReLU-LayerNorm layers.

    Args:
        embed_dims: Dimension of embeddings
        in_loops: Number of input transformation loops
        out_loops: Number of output transformation loops
        input_dims: Input dimensions (default: embed_dims)

    Returns:
        List of nn.Module layers
    """
    if input_dims is None:
        input_dims = embed_dims

    layers = []

    # Input loops
    for i in range(in_loops):
        if i == 0:
            layers.append(nn.Linear(input_dims, embed_dims))
        else:
            layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(embed_dims))

    # Output loops
    for i in range(out_loops):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(embed_dims))

    return layers


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability.

    Args:
        prior_prob: Prior probability

    Returns:
        Bias initialization value
    """
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize module parameters using Xavier initialization.

    Args:
        module: nn.Module to initialize
        gain: Scaling factor
        bias: Bias value
        distribution: 'normal' or 'uniform'
    """
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class BaseTargetWithDenoising(nn.Module):
    """Base class for target assignment with denoising support.

    This class provides basic denoising functionality for training.
    """

    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        super().__init__()
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None