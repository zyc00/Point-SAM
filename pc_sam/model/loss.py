from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-3,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape, [B, ..., N].
                The (probability) predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        reduction: ``'none'`` | ``'mean'`` | ``'sum'``
        eps: A small epsilon value to avoid division by zero.

    Returns:
        torch.Tensor: If reduction is 'none', then the shape is [B, ...]. Otherwise, a scalar is returned.

    References:
        https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
        https://github.com/UX-Decoder/Semantic-SAM/blob/3d6a43a0f8e77167c0013d14067933a78e2d1f5a/semantic_sam/modules/criterion_interactive_many_to_many.py#L57
        https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/losses/dice_loss.py#L9
    """
    assert inputs.shape == targets.shape, (inputs.shape, targets.shape)
    assert inputs.dtype == targets.dtype, (inputs.dtype, targets.dtype)

    numerator = 2 * (inputs * targets).sum(-1)
    # NOTE: If target is binary, target equals to target.square()
    denominator = inputs.square().sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def compute_mask_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_weight_dice: float = 2
):
    """Loss for mask prediction.

    Args:
        logits: A float tensor of shape [B, C, N]. Multi-mask predicted logits.
        labels: A float tensor of shape [B, N]. Ground-truth binary masks.

    Returns:
        torch.Tensor: [B, C]. Mask loss
    """
    assert logits.dim() == 3, logits.shape
    _labels = labels.unsqueeze(1).expand_as(logits)
    _labels = _labels.to(dtype=logits.dtype)
    # loss_ce = F.binary_cross_entropy_with_logits(logits, _labels, reduction="none")
    loss_ce = sigmoid_focal_loss(logits, _labels, alpha=-1, reduction="none")
    loss_dice = dice_loss(logits.sigmoid(), _labels, reduction="none")
    loss = loss_ce.mean(-1) + loss_weight_dice * loss_dice
    return loss


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = None):
    """Compute intersection-over-union (IoU).

    Args:
        logits: A float tensor of shape [..., N]. Multi-mask predicted logits.
        targets: A bool tensor of shape [..., N]. Ground-truth binary masks.
        threshold: A float value for thresholding predictions.
            If None, use the default threshold of 0.5.

    Returns:
        torch.Tensor: [...]. IoU scores
    """
    assert logits.shape == targets.shape, (logits.shape, targets.shape)
    assert targets.dtype == torch.bool, targets.dtype
    if threshold is None:
        preds = logits > 0
    else:
        preds = logits.sigmoid() > threshold
    return (preds & targets).sum(-1) / (preds | targets).sum(-1)


@torch.jit.script
def compute_jaccard(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-3):
    assert logits.shape == targets.shape, (logits.shape, targets.shape)
    probs = logits.sigmoid()
    numerator = (probs * targets).sum(-1)
    denominator = (probs.square() + targets.square()).sum(-1) - numerator
    return (numerator + eps) / (denominator + eps)


class Criterion(nn.Module):
    def __init__(self, use_soft_iou=False):
        super().__init__()
        self.use_soft_iou = use_soft_iou

    def forward(self, outputs: List[Dict[str, torch.Tensor]], gt_masks):
        # gt_mask: [B*M, N]
        # Follow the "Making the model ambiguity-aware" in Appendix A of SAM.
        # Multimask is only enabled with more than one prompt.
        losses = []
        aux_outputs = []
        for i, output in enumerate(outputs):
            masks = output["masks"]  # [B*M, C, N]
            iou_preds = output["iou_preds"]  # [B*M, C]

            loss_mask = compute_mask_loss(masks, gt_masks)  # [B*M,C]
            if i == 0:
                loss_mask, min_loss_idx = loss_mask.min(dim=1)  # [B*M]
                batch_idx = torch.arange(min_loss_idx.shape[0])
                best_masks = masks[batch_idx, min_loss_idx]  # [B*M, N]
                iou_preds = iou_preds[batch_idx, min_loss_idx]  # [B*M]
            else:
                best_masks = masks.squeeze(1)
                iou_preds = iou_preds.squeeze(1)
            loss_mask = loss_mask.mean()

            iou = compute_iou(best_masks, gt_masks)  # [B*M]
            if self.use_soft_iou:
                with torch.no_grad():
                    soft_iou = compute_jaccard(
                        best_masks, gt_masks.to(dtype=best_masks.dtype)
                    )
                loss_iou = F.mse_loss(soft_iou, iou_preds)
            else:
                loss_iou = F.mse_loss(iou, iou_preds)

            losses.append(loss_iou + loss_mask)
            # losses.append(loss_mask)
            aux_outputs.append(
                dict(
                    iou=iou,
                    best_masks=best_masks,
                    loss_mask=loss_mask,
                    loss_iou=loss_iou,
                )
            )

        loss = torch.stack(losses).mean()
        return loss, aux_outputs
