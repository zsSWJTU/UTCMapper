"""
CCG Loss: Consistent and Confident Guidance Loss

Updated version:
- Consistent region uses Dice loss
- Inconsistent region uses BootLoss
- Cleaner masking logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Dice Loss
# =========================================================
class BinaryDiceLoss(nn.Module):
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
# 2. Boot Loss
# =========================================================
class BootLoss(nn.Module):
    def __init__(
        self,
        reduce=True,
        as_pseudo_label=True,
        ignore_index=255,
    ):
        super().__init__()

        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label
        self.ignore_index = ignore_index

    def forward(
        self,
        y_pred,
        y,
    ):
        """
        y_pred: [B, C, H, W]
        y     : [B, H, W]

        Note:
            BootLoss itself does not use y as class index.
            y is only used to decide which pixels are valid.
        """

        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0, :, :]

        valid_mask = (
            y != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(y, dtype=torch.bool)
        )

        if valid_mask.sum() == 0:
            return y_pred.sum() * 0.0

        y_pred_a = (
            y_pred.detach()
            if self.as_pseudo_label
            else y_pred
        )

        boot_loss = -torch.sum(
            F.softmax(y_pred_a, dim=1)
            * F.log_softmax(y_pred, dim=1),
            dim=1,
        )

        if self.reduce:
            return boot_loss[valid_mask].mean()

        out = torch.zeros_like(
            boot_loss
        )
        out[valid_mask] = boot_loss[valid_mask]

        return out

# =========================================================
# 3. CCGLoss
# =========================================================
class CCGLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        ignore_index=255,
        fn_quantile=0.5,
        fp_quantile=0.5,
    ):
        super().__init__()

        self.ignore_index = ignore_index

        self.dice = BinaryDiceLoss()
        self.boot = BootLoss(
            ignore_index=ignore_index,
        )
        self.alpha = alpha
        self.fn_quantile = self._validate_quantile("fn_quantile", fn_quantile)
        self.fp_quantile = self._validate_quantile("fp_quantile", fp_quantile)

    @staticmethod
    def _validate_quantile(name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return value

    # -----------------------------
    # confidence
    # -----------------------------
    def calculate_confidence(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        top2, _ = torch.topk(probs, k=2, dim=1)

        return (top2[:, 0] - top2[:, 1]) / (top2[:, 0] + top2[:, 1] + 1e-8)

    # -----------------------------
    # inconsistent mask
    # -----------------------------
    def find_inconsistent_mask(
            self,
            outputs,
            target,
    ):

        confidence = self.calculate_confidence(
            outputs
        )

        pred_labels = torch.argmax(
            outputs,
            dim=1,
        )

        valid_mask = target != self.ignore_index

        if outputs.shape[1] != 2:
            raise ValueError(
                "FN/FP-specific CCG selection requires exactly two classes, "
                f"got {outputs.shape[1]}."
            )

        false_negative = valid_mask & (target == 1) & (pred_labels == 0)
        false_positive = valid_mask & (target == 0) & (pred_labels == 1)

        def select(error_mask, quantile):
            values = confidence[error_mask]
            if values.numel() == 0:
                return error_mask
            threshold = torch.quantile(values.detach().float(), quantile)
            return error_mask & (confidence >= threshold)

        # FN and FP often represent different annotation errors. Selecting
        # them within their own confidence distributions prevents a dominant
        # error type from setting the threshold for the other one.
        return (
            select(false_negative, self.fn_quantile)
            | select(false_positive, self.fp_quantile)
        )

    # -----------------------------
    # forward
    # -----------------------------
    def forward(
            self,
            outputs,
            target,
    ):

        """
        outputs: [B, C, H, W]
        target : [B, H, W]
                 valid classes: 0 ~ C-1
                 ignore_index : self.ignore_index, usually 255
        """

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0, :, :]

        target = target.long()

        num_classes = outputs.shape[1]

        valid_mask = target != self.ignore_index

        # -----------------------------------------------------
        # If all pixels are ignored, return zero loss safely
        # -----------------------------------------------------
        if valid_mask.sum() == 0:
            zero_loss = outputs.sum() * 0.0

            return {
                "consistent_loss": zero_loss,
                "inconsistent_loss": zero_loss,
            }

        # -----------------------------------------------------
        # safe_target is only used for one_hot / indexing.
        # Ignore pixels are temporarily set to 0 to avoid
        # out-of-bound errors.
        # -----------------------------------------------------
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        # Optional safety check
        invalid_mask = (
                valid_mask
                & (
                        (safe_target < 0)
                        | (safe_target >= num_classes)
                )
        )

        if invalid_mask.any():
            bad_values = torch.unique(
                target[invalid_mask].detach().cpu()
            )

            raise RuntimeError(
                f"Invalid target values detected in CCGLoss: "
                f"{bad_values.tolist()} | "
                f"Allowed: 0~{num_classes - 1}, "
                f"ignore_index={self.ignore_index}"
            )

        # =========================
        # mask generation
        # =========================
        with torch.no_grad():

            mask_inconsistent = self.find_inconsistent_mask(
                outputs,
                target,
            )

            mask_consistent = (
                    (~mask_inconsistent)
                    & valid_mask
            )

        # =========================
        # 1. Consistent loss
        # =========================

        probs = torch.softmax(
            outputs,
            dim=1,
        )

        one_hot_target = F.one_hot(
            safe_target,
            num_classes=num_classes,
        ).float().permute(
            0,
            3,
            1,
            2,
        )

        mask_consistent_2c = mask_consistent.unsqueeze(1).expand(
            -1,
            num_classes,
            -1,
            -1,
        ).float()

        if mask_consistent.sum() == 0:
            consistent_loss = outputs.sum() * 0.0
        else:
            consistent_loss = self.dice(
                probs * mask_consistent_2c,
                one_hot_target * mask_consistent_2c,
            )

        # =========================
        # 2. Inconsistent loss
        # =========================
        target_inconsistent = target.clone()

        # Ignore consistent pixels and original ignore pixels
        target_inconsistent[~mask_inconsistent] = self.ignore_index

        if mask_inconsistent.sum() == 0:
            inconsistent_loss = outputs.sum() * 0.0
        else:
            inconsistent_loss = self.boot(
                outputs,
                target_inconsistent,
            )

        # =========================
        # output
        # =========================
        return {
            "consistent_loss": consistent_loss,
            "inconsistent_loss": self.alpha * inconsistent_loss,
        }


# =========================================================
# 4. Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 2, 256, 256

    outputs = torch.randn(B, C, H, W).to(device)
    target = torch.randint(0, C, (B, H, W)).to(device)

    # ---- Dice mode ----
    loss_fn_dice = CCGLoss(alpha=0.5).to(device)
    loss_dice = loss_fn_dice(outputs, target)

    print("Dice mode:")
    print(loss_dice)

