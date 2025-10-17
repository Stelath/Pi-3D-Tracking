"""Detection3D Head module for 3D object detection.

This module implements a transformer-based detection head that connects to the Pi3 backbone.
It follows the architecture pattern of CameraHead but adapted for 3D object detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
)


class Detection3DHead(nn.Module):
    """3D object detection head that processes multi-view features from Pi3.

    Args:
        pi3_embed_dim: Dimension of features from Pi3 decoder (default: 2048 = 2*1024)
        embed_dim: Internal embedding dimension (default: 256)
        num_queries: Number of detection queries (default: 300)
        num_classes: Number of object classes (no default - must specify)
        box_output_dim: Box output dimension (default: 10 for [x,y,z,log_w,log_h,log_l,sin_yaw,cos_yaw,vx,vy])
        num_decoder_layers: Number of transformer decoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        ffn_dim: Feed-forward network dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
        with_quality_estimation: Enable centerness/yawness prediction (default: True)
        num_keypoints: Number of keypoints per query for feature sampling (default: 1)
    """

    def __init__(
        self,
        pi3_embed_dim=2048,
        embed_dim=256,
        num_queries=300,
        num_classes=10,
        box_output_dim=10,
        num_decoder_layers=6,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        with_quality_estimation=True,
        num_keypoints=1,
    ):
        super().__init__()

        self.pi3_embed_dim = pi3_embed_dim
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.box_output_dim = box_output_dim
        self.num_decoder_layers = num_decoder_layers
        self.with_quality_estimation = with_quality_estimation

        # Project Pi3 features to detection embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(pi3_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Learnable anchor boxes (initialized as zeros, will be refined)
        # Format: [x, y, z, log_w, log_h, log_l, sin_yaw, cos_yaw, vx, vy]
        self.anchor_embed = nn.Embedding(num_queries, box_output_dim)
        nn.init.zeros_(self.anchor_embed.weight)

        # Box encoder: encodes anchor boxes to embeddings
        self.box_encoder = SparseBox3DEncoder(
            embed_dims=embed_dim,
            vel_dims=2,  # vx, vy (no vz for ground-based objects)
            mode="add",
            output_fc=True,
            in_loops=1,
            out_loops=2,
        )

        # Keypoint generator for multi-scale feature sampling
        self.keypoint_generator = SparseBox3DKeyPointsGenerator(
            embed_dims=embed_dim,
            num_learnable_pts=num_keypoints - 1 if num_keypoints > 0 else 0,
            fix_scale=((0.0, 0.0, 0.0),) if num_keypoints > 0 else None,
        )

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            ) for _ in range(num_decoder_layers)
        ])

        # Box refinement modules (one per decoder layer)
        self.refinement_modules = nn.ModuleList([
            SparseBox3DRefinementModule(
                embed_dims=embed_dim,
                output_dim=box_output_dim,
                num_cls=num_classes,
                normalize_yaw=True,
                refine_yaw=True,
                with_cls_branch=True,
                with_quality_estimation=with_quality_estimation,
            ) for _ in range(num_decoder_layers)
        ])

        # Initialize refinement module weights
        for module in self.refinement_modules:
            module.init_weight()
        self.keypoint_generator.init_weight()

    def forward(self, pi3_features, patch_h, patch_w):
        """Forward pass of the detection head.

        Args:
            pi3_features: Features from Pi3 decoder [B*N, num_patches, pi3_embed_dim]
                where B is batch size, N is number of cameras
            patch_h: Height of feature patches
            patch_w: Width of feature patches

        Returns:
            Dictionary with:
                - box_pred: List of box predictions per layer [B*N, num_queries, box_output_dim]
                - cls_pred: List of class predictions per layer [B*N, num_queries, num_classes]
                - quality_pred: List of quality predictions per layer [B*N, num_queries, 2] (if enabled)
        """
        BN, num_patches, _ = pi3_features.shape

        # Project Pi3 features to detection embedding space
        # [B*N, num_patches, embed_dim]
        memory = self.input_proj(pi3_features)

        # Initialize query embeddings
        # [num_queries, embed_dim] -> [B*N, num_queries, embed_dim]
        query = self.query_embed.weight.unsqueeze(0).expand(BN, -1, -1)

        # Initialize anchor boxes
        # [num_queries, box_output_dim] -> [B*N, num_queries, box_output_dim]
        anchor = self.anchor_embed.weight.unsqueeze(0).expand(BN, -1, -1)

        # Storage for predictions at each layer (for deep supervision)
        box_preds = []
        cls_preds = []
        quality_preds = []

        # Iterative refinement through decoder layers
        for layer_idx, (decoder_layer, refinement_module) in enumerate(
            zip(self.decoder_layers, self.refinement_modules)
        ):
            # Encode current anchor boxes to embeddings
            anchor_embed = self.box_encoder(anchor)

            # Generate keypoints from current anchors for feature sampling
            # (This could be used for deformable attention in future - for now just pass)
            keypoints = self.keypoint_generator(anchor, query)

            # Transformer decoder: query attends to Pi3 features
            query = decoder_layer(
                query=query,
                key=memory,
                value=memory,
            )

            # Refine boxes and predict classes
            anchor, cls_score, quality_score = refinement_module(
                instance_feature=query,
                anchor=anchor,
                anchor_embed=anchor_embed,
                time_interval=1.0,
                return_cls=True,
            )

            # Store predictions for this layer
            box_preds.append(anchor)
            cls_preds.append(cls_score)
            if quality_score is not None:
                quality_preds.append(quality_score)

        output = {
            'box_pred': box_preds,
            'cls_pred': cls_preds,
        }

        if self.with_quality_estimation and len(quality_preds) > 0:
            output['quality_pred'] = quality_preds

        return output


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
    """

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        """Forward pass.

        Args:
            query: Query tensor [B, num_queries, embed_dim]
            key: Key tensor [B, num_keys, embed_dim]
            value: Value tensor [B, num_keys, embed_dim]
            query_mask: Optional attention mask for queries
            key_mask: Optional attention mask for keys

        Returns:
            Updated query tensor [B, num_queries, embed_dim]
        """
        # Self-attention
        q = query
        attn_output, _ = self.self_attn(
            query=q,
            key=q,
            value=q,
            attn_mask=query_mask,
        )
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)

        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=key_mask,
        )
        query = query + self.dropout2(attn_output)
        query = self.norm2(query)

        # Feed-forward
        ffn_output = self.ffn(query)
        query = query + self.dropout3(ffn_output)
        query = self.norm3(query)

        return query
