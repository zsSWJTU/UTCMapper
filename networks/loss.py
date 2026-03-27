"""
CCG Loss: Consistent and Confident Guidance Loss

The loss separates pixels into:
    - Consistent regions: predicted labels align with coarse supervision → Dice loss
    - Inconsistent regions: predicted labels conflict with coarse labels → BootLoss (self-supervised)

Key components:
    1. BinaryDiceLoss: Measures overlap for consistent regions.
    2. BootLoss: Encourages confident predictions for uncertain regions.
    3. CCGLoss: Combines both losses using confidence-guided masking.

Author: Shuang Zhang
License: Apache-2.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Dice Loss (for consistent regions)
# =========================================================
class BinaryDiceLoss(nn.Module):
    """
    Dice loss for remaining set (R)
    Applies to pixels where coarse label is considered reliable.
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(predict * target, dim=1) + self.smooth
        denominator = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - intersection / denominator

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unexpected reduction: {self.reduction}")


# =========================================================
# 2. Boot Loss (for inconsistent regions)
# =========================================================
class BootLoss(nn.Module):
    """
    BootLoss for filtered set (F)
    Applies to pixels where coarse label may contain errors.
    Encourages the model to maintain high confidence in predictions
    and reduces impact of potentially incorrect coarse labels.
    """
    def __init__(self, reduce=True, as_pseudo_label=True, ignore_index=-1):
        super().__init__()
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label
        self.ignore_index = ignore_index

    def forward(self, y_pred, y):
        mask = (y != self.ignore_index).float() if self.ignore_index is not None else torch.ones_like(y)
        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        boot_loss = -torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1) * mask
        return boot_loss.mean() if self.reduce else boot_loss


# =========================================================
# 3. CCGLoss (main loss for UTCMapper)
# =========================================================
class CCGLoss(nn.Module):
    """
    CCGLoss implements Conflict and Confidence-Guided loss :
    1. Compute pixel-wise confidence
    2. Identify potential error pixels: conflicting & confident → filtered set F
    3. Partition remaining pixels → remaining set R
    4. Apply Dice loss to R (supervised) and BootLoss to F (self-supervised)
    """
    def __init__(self, alpha=1):
        super().__init__()
        self.dice = BinaryDiceLoss()
        self.boot = BootLoss()
        self.alpha = alpha

    def calculate_confidence(self, outputs):
        """
        Pixel-wise confidence: difference between top two softmax probabilities.
        """
        probs = torch.softmax(outputs, dim=1)
        top2, _ = torch.topk(probs, k=2, dim=1)
        return (top2[:, 0] - top2[:, 1]) / (top2[:, 0] + top2[:, 1] + 1e-8)

    def find_inconsistent_mask(self, outputs, target):
        """
        Identify pixels where prediction disagrees with target but confidence is high.
        """
        confidence = self.calculate_confidence(outputs)
        pred_labels = torch.argmax(outputs, dim=1)
        mismatch = (pred_labels != target)
        threshold = torch.mean(confidence).item()
        inconsistent_mask = mismatch & (confidence > threshold)
        return inconsistent_mask

    def forward(self, outputs, target):
        # generate masks
        with torch.no_grad():
            mask_inconsistent = self.find_inconsistent_mask(outputs, target)
            mask_inconsistent_2c = mask_inconsistent.unsqueeze(1).repeat(1, outputs.shape[1], 1, 1)
            mask_consistent_2c = ~mask_inconsistent_2c

        # one-hot encoding
        one_hot_target = F.one_hot(target, num_classes=outputs.shape[1]).float().permute(0, 3, 1, 2)

        # consistent region Dice loss
        consistent_loss = self.dice(mask_consistent_2c * outputs, mask_consistent_2c * one_hot_target)

        # inconsistent region BootLoss
        target_inconsistent = target.clone()
        target_inconsistent[~mask_inconsistent] = -1
        inconsistent_loss = self.boot(outputs, target_inconsistent)

        return {
            'consistent_loss': consistent_loss,
            'inconsistent_loss': self.alpha * inconsistent_loss
        }
