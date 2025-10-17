import torch
import torch.nn as nn

from modules.constants import X, Y, Z, SIN_YAW, COS_YAW, CNS, YNS


class SparseBox3DLoss(nn.Module):
    """Loss module for 3D box detection.

    Computes losses for box regression, centerness, and yawness predictions.
    """

    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ):
        """Initialize loss module.

        Args:
            loss_box: Loss module for box regression (e.g., nn.L1Loss, nn.SmoothL1Loss)
            loss_centerness: Optional loss module for centerness prediction
            loss_yawness: Optional loss module for yawness prediction
            cls_allow_reverse: Optional list of class IDs that allow reversed directions
        """
        super().__init__()

        self.loss_box = loss_box
        self.loss_cns = loss_centerness
        self.loss_yns = loss_yawness
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        """Compute losses.

        Args:
            box: Predicted boxes
            box_target: Target boxes
            weight: Optional per-element weights
            avg_factor: Optional normalization factor
            suffix: Optional suffix for loss names
            quality: Optional quality predictions (centerness, yawness)
            cls_target: Optional class targets for reverse direction handling

        Returns:
            Dictionary of losses
        """
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}

        # Compute box loss
        if weight is not None:
            box_loss = self.loss_box(box * weight, box_target * weight)
        else:
            box_loss = self.loss_box(box, box_target)

        if avg_factor is not None:
            box_loss = box_loss.sum() / avg_factor
        else:
            box_loss = box_loss.mean()

        output[f"loss_box{suffix}"] = box_loss

        # Compute quality losses if provided
        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()

            # Centerness target: based on distance between prediction and target
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)

            cns_loss = self.loss_cns(cns, cns_target)
            if avg_factor is not None:
                cns_loss = cns_loss.sum() / avg_factor
            else:
                cns_loss = cns_loss.mean()
            output[f"loss_cns{suffix}"] = cns_loss

            # Yawness target: whether yaw directions match
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()

            yns_loss = self.loss_yns(yns, yns_target)
            if avg_factor is not None:
                yns_loss = yns_loss.sum() / avg_factor
            else:
                yns_loss = yns_loss.mean()
            output[f"loss_yns{suffix}"] = yns_loss

        return output