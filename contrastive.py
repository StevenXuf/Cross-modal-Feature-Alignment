import torch
import torch.nn as nn

class CustomContrastiveLossWithMargin(nn.Module):
    def __init__(self, margin=0.2, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, outputs, targets=None):
        """
        outputs: [B, B]
            outputs[i, j] = predicted similarity between text_i and image_j
        """

        B = outputs.size(0)
        device = outputs.device

        pos = outputs.diag()  # [B]

        # text-to-image ranking: row-wise
        loss_t2i = self.margin + outputs - pos[:, None]

        # image-to-text ranking: column-wise
        loss_i2t = self.margin + outputs - pos[None, :]

        eye = torch.eye(B, device=device, dtype=torch.bool)

        loss_t2i = loss_t2i.masked_fill(eye, 0.0)
        loss_i2t = loss_i2t.masked_fill(eye, 0.0)

        loss_t2i = torch.clamp(loss_t2i, min=0.0)
        loss_i2t = torch.clamp(loss_i2t, min=0.0)

        loss = 0.5 * (loss_t2i + loss_i2t)

        if self.reduction == "mean":
            return loss.sum() / (B * (B - 1))
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class CustomContrastiveLossBalanced(nn.Module):
    def __init__(self, reduction="balanced_mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, targets=None):
        """
        outputs: [B, B]
            outputs[i, j] = d_hat_ij = Z(cat(x_i, y_j))

        targets: unused; positive pairs are diagonal.
        """
        B = outputs.size(0)
        device = outputs.device

        I = torch.eye(B, device=device, dtype=torch.bool)

        pos_scores = outputs[I]      # d_hat_ii
        neg_scores = outputs[~I]     # d_hat_ij, i != j

        pos_loss = 0.5 * (1.0 - pos_scores).pow(2)
        neg_loss = 0.5 * (neg_scores + 1.0).pow(2)

        if self.reduction == "balanced_mean":
            return pos_loss.mean() + neg_loss.mean()

        elif self.reduction == "mean":
            # Original matrix mean, but this is usually bad because negatives dominate.
            loss_matrix = torch.zeros_like(outputs)
            loss_matrix[I] = pos_loss
            loss_matrix[~I] = neg_loss
            return loss_matrix.mean()

        elif self.reduction == "sum":
            return pos_loss.sum() + neg_loss.sum()

        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")