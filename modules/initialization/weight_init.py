"""Weight initialization utilities for neural networks."""
import math
import torch.nn as nn


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability.

    This is commonly used for classification layers where you want to
    initialize the bias based on the expected prior probability of a class.
    Uses the inverse sigmoid (logit) function.

    Args:
        prior_prob: Prior probability (between 0 and 1)

    Returns:
        Bias initialization value

    Example:
        >>> bias = bias_init_with_prob(0.01)  # For rare objects
        >>> nn.init.constant_(classifier.bias, bias)
    """
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize module parameters using Xavier initialization.

    Xavier initialization helps maintain the variance of activations and
    gradients across layers, which is important for deep networks.

    Args:
        module: nn.Module to initialize (must have weight/bias attributes)
        gain: Scaling factor for the weights
        bias: Constant value for bias initialization
        distribution: 'normal' for Gaussian or 'uniform' for uniform distribution

    Example:
        >>> linear = nn.Linear(256, 512)
        >>> xavier_init(linear, gain=1, bias=0, distribution='normal')
    """
    assert distribution in ['uniform', 'normal'], \
        f"distribution must be 'uniform' or 'normal', got {distribution}"

    # Initialize weights
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)

    # Initialize bias
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


__all__ = ['bias_init_with_prob', 'xavier_init']
