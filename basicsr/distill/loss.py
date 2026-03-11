"""
Knowledge distillation loss for LUMEN.

Directly adapted from MambaLiteSR (arXiv:2502.14090):

    L = alpha * L1(student_output, teacher_output)
      + (1 - alpha) * L1(student_output, ground_truth)

MambaLiteSR found alpha=0.8 outperforms alpha=0.5 (naive 50/50 split).
The higher alpha means the student leans more on the teacher's soft outputs,
which act as dense supervision even in regions where GT is ambiguous.

For LUMEN we also support feature-level distillation (intermediate features)
as an optional add-on, which helps when teacher and student have different depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DistillationLoss(nn.Module):
    """Output-level knowledge distillation loss (MambaLiteSR style).

    Args:
        alpha: Weight on teacher supervision. MambaLiteSR used 0.8.
               alpha=1.0 → pure teacher KD (ignores GT)
               alpha=0.0 → standard supervised loss (ignores teacher)
        loss_type: 'l1' (default, used by MambaLiteSR) or 'l2' or 'charbonnier'
        eps: Epsilon for Charbonnier loss stability.
    """

    def __init__(self, alpha: float = 0.8, loss_type: str = 'l1', eps: float = 1e-3):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha
        self.loss_type = loss_type
        self.eps = eps

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'charbonnier':
            diff = pred - target
            return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple:
        """
        Args:
            student_out: Student model output (B, C, H, W).
            teacher_out: Teacher model output, same shape (no grad needed).
            gt: Ground-truth HR image, same shape.

        Returns:
            (total_loss, loss_kd, loss_gt): tuple of scalar tensors.
        """
        loss_kd = self._loss(student_out, teacher_out.detach())
        loss_gt = self._loss(student_out, gt)
        total = self.alpha * loss_kd + (1.0 - self.alpha) * loss_gt
        return total, loss_kd, loss_gt
