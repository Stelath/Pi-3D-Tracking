"""Pi3Detection3D - Combined model with Pi3 backbone and Detection3D head.

This module implements the main detection model that uses Pi3 as a frozen backbone
and adds a trainable Detection3D head for 3D object detection.
"""
import torch
import torch.nn as nn
from pathlib import Path

from model.pi3.models.pi3 import Pi3
from model.detection3d.detection3d_head import Detection3DHead


class Pi3Detection3D(nn.Module):
    """3D object detection model with Pi3 backbone and Detection3D head.

    The Pi3 backbone is loaded from a pretrained checkpoint and frozen during training.
    Only the Detection3D head parameters are trainable.

    Args:
        pi3_checkpoint: Path to pretrained Pi3 checkpoint (safetensors or pt format)
        num_classes: Number of object classes to detect (required)
        num_queries: Number of detection queries (default: 300)
        embed_dim: Internal embedding dimension for detection head (default: 256)
        num_decoder_layers: Number of transformer decoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        box_output_dim: Box output dimension (default: 10)
        with_quality_estimation: Enable centerness/yawness prediction (default: True)
        freeze_pi3: Whether to freeze Pi3 backbone (default: True)
    """

    def __init__(
        self,
        pi3_checkpoint=None,
        num_classes=10,
        num_queries=300,
        embed_dim=256,
        num_decoder_layers=6,
        num_heads=8,
        box_output_dim=10,
        with_quality_estimation=True,
        freeze_pi3=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.freeze_pi3 = freeze_pi3

        # ----------------------
        #  Load Pi3 Backbone
        # ----------------------
        print("Loading Pi3 backbone...")
        if pi3_checkpoint is not None and Path(pi3_checkpoint).exists():
            # Load Pi3 model structure
            self.pi3 = Pi3(pos_type='rope100', decoder_size='large')

            # Load pretrained weights
            if str(pi3_checkpoint).endswith('.safetensors'):
                from safetensors.torch import load_file
                weights = load_file(pi3_checkpoint)
            else:
                weights = torch.load(pi3_checkpoint, map_location='cpu', weights_only=False)

            self.pi3.load_state_dict(weights, strict=True)
            print(f"Loaded Pi3 weights from {pi3_checkpoint}")
        else:
            # Load from HuggingFace hub
            print("Loading Pi3 from HuggingFace (yyfz233/Pi3)...")
            self.pi3 = Pi3.from_pretrained("yyfz233/Pi3")

        # Freeze Pi3 backbone if requested
        if freeze_pi3:
            print("Freezing Pi3 backbone parameters...")
            for param in self.pi3.parameters():
                param.requires_grad = False
            self.pi3.eval()

        # Get Pi3 decoder output dimension
        # Pi3 decoder outputs concatenated features from last 2 layers
        pi3_embed_dim = self.pi3.dec_embed_dim * 2  # 2048 for large decoder

        # ----------------------
        #  Detection3D Head
        # ----------------------
        print("Initializing Detection3D head...")
        self.detection_head = Detection3DHead(
            pi3_embed_dim=pi3_embed_dim,
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            box_output_dim=box_output_dim,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            with_quality_estimation=with_quality_estimation,
        )

        print(f"Pi3Detection3D initialized:")
        print(f"  - Classes: {num_classes}")
        print(f"  - Queries: {num_queries}")
        print(f"  - Decoder layers: {num_decoder_layers}")
        print(f"  - Pi3 frozen: {freeze_pi3}")

    def forward(self, imgs):
        """Forward pass.

        Args:
            imgs: Input images [B, N, 3, H, W]
                where B is batch size, N is number of cameras

        Returns:
            Dictionary with:
                - box_pred: List of box predictions per layer [B*N, num_queries, box_output_dim]
                - cls_pred: List of class predictions per layer [B*N, num_queries, num_classes]
                - quality_pred: List of quality predictions per layer [B*N, num_queries, 2] (if enabled)
        """
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # Extract features from Pi3 backbone
        if self.freeze_pi3:
            with torch.no_grad():
                pi3_features = self._extract_pi3_features(imgs)
        else:
            pi3_features = self._extract_pi3_features(imgs)

        # Pass through detection head
        # Use AMP for detection head (following Pi3 pattern)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Convert to float32 for detection head (stable training)
            pi3_features = pi3_features.float()
            detections = self.detection_head(pi3_features, patch_h, patch_w)

        return detections

    def _extract_pi3_features(self, imgs):
        """Extract features from Pi3 backbone.

        Args:
            imgs: Input images [B, N, 3, H, W]

        Returns:
            Concatenated features from last 2 decoder layers [B*N, num_patches+patch_start_idx, 2*dec_embed_dim]
        """
        # Normalize images (Pi3 uses ImageNet normalization)
        imgs = (imgs - self.pi3.image_mean) / self.pi3.image_std

        B, N, _, H, W = imgs.shape

        # Encode with DINOv2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.pi3.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # Pass through Pi3 decoder
        # This returns concatenated features from last 2 layers
        features, pos = self.pi3.decode(hidden, N, H, W)

        return features

    def train(self, mode=True):
        """Override train method to keep Pi3 in eval mode if frozen."""
        super().train(mode)
        if self.freeze_pi3:
            self.pi3.eval()
        return self

    def get_trainable_parameters(self):
        """Get only trainable parameters (excludes frozen Pi3)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
        }


class Pi3Detection3DWithAuxHeads(Pi3Detection3D):
    """Extended model that keeps Pi3's auxiliary heads for multi-task learning.

    This variant keeps Pi3's original heads (points, confidence, camera) active
    for auxiliary supervision during training, which may improve feature quality.

    Args:
        Same as Pi3Detection3D, plus:
        aux_loss_weights: Dictionary of loss weights for auxiliary tasks
            e.g., {'points': 0.1, 'conf': 0.1, 'camera': 0.1}
    """

    def __init__(
        self,
        aux_loss_weights=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.aux_loss_weights = aux_loss_weights or {}

    def forward(self, imgs):
        """Forward pass with auxiliary outputs.

        Args:
            imgs: Input images [B, N, 3, H, W]

        Returns:
            Dictionary with:
                - box_pred: Detection box predictions
                - cls_pred: Detection class predictions
                - quality_pred: Detection quality predictions
                - aux_outputs: Dictionary with Pi3 auxiliary outputs
                    - points: 3D points
                    - local_points: Local 3D points
                    - conf: Confidence maps
                    - camera_poses: Camera poses
        """
        B, N, _, H, W = imgs.shape

        # Get detection outputs
        detections = super().forward(imgs)

        # Get Pi3 auxiliary outputs if requested
        if self.training and self.aux_loss_weights:
            with torch.no_grad():
                pi3_outputs = self.pi3(imgs)

            detections['aux_outputs'] = {
                'points': pi3_outputs['points'],
                'local_points': pi3_outputs['local_points'],
                'conf': pi3_outputs['conf'],
                'camera_poses': pi3_outputs['camera_poses'],
            }

        return detections
