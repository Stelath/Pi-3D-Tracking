"""Neural network operations and layer utilities."""
import torch
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable parameter.
    Replaces mmcv.cnn.Scale.

    Args:
        scale: Initial scale value(s). Can be a scalar, list, or tuple.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        if isinstance(scale, (list, tuple)):
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        """Scale input by learnable parameter.

        Args:
            x: Input tensor

        Returns:
            Scaled tensor
        """
        return x * self.scale


def linear_relu_ln(embed_dims, in_loops=1, out_loops=1, input_dims=None):
    """Create a sequence of Linear-ReLU-LayerNorm layers.

    This is a common pattern for building MLP blocks in transformers.

    Args:
        embed_dims: Dimension of embeddings
        in_loops: Number of input transformation loops
        out_loops: Number of output transformation loops
        input_dims: Input dimensions (default: embed_dims)

    Returns:
        List of nn.Module layers that can be unpacked into nn.Sequential
    """
    if input_dims is None:
        input_dims = embed_dims

    layers = []

    # Input loops: transform input_dims to embed_dims
    for i in range(in_loops):
        if i == 0:
            layers.append(nn.Linear(input_dims, embed_dims))
        else:
            layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(embed_dims))

    # Output loops: keep at embed_dims
    for i in range(out_loops):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(embed_dims))

    return layers


__all__ = ['Scale', 'linear_relu_ln']
