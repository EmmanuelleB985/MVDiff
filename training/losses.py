"""
Loss functions for MVDiff training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Simple MSE loss for diffusion model training.
    Can be extended with additional terms.
    """

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        """Compute loss between predicted and target noise"""
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MultiViewConsistencyLoss(nn.Module):
    """
    Additional loss for enforcing multi-view consistency.
    Uses epipolar constraints and photometric consistency.
    """

    def __init__(self, weight_epipolar=0.1, weight_photo=0.1):
        super().__init__()
        self.weight_epipolar = weight_epipolar
        self.weight_photo = weight_photo

    def epipolar_loss(self, pts1, pts2, F):
        """
        Compute epipolar constraint loss.
        pts2^T * F * pts1 should be close to 0.
        """
        # Convert points to homogeneous coordinates
        ones = torch.ones_like(pts1[..., :1])
        pts1_h = torch.cat([pts1, ones], dim=-1)
        pts2_h = torch.cat([pts2, ones], dim=-1)

        # Compute epipolar constraint
        Fp1 = torch.matmul(F, pts1_h.transpose(-2, -1))
        p2Fp1 = torch.sum(pts2_h.unsqueeze(-2) * Fp1.transpose(-2, -1), dim=-1)

        return torch.abs(p2Fp1).mean()

    def photometric_loss(self, img1, img2, mask=None):
        """
        Compute photometric consistency loss between views.
        """
        diff = (img1 - img2).abs()

        if mask is not None:
            diff = diff * mask
            return diff.sum() / (mask.sum() + 1e-8)
        else:
            return diff.mean()

    def forward(self, views, fundamental_matrices, correspondences=None):
        """
        Compute multi-view consistency loss.

        Args:
            views: [B, N, C, H, W] - multi-view images
            fundamental_matrices: [B, N, N, 3, 3] - fundamental matrices
            correspondences: optional point correspondences
        """
        B, N = views.shape[:2]
        total_loss = 0

        # Pairwise consistency
        for i in range(N):
            for j in range(i + 1, N):
                view_i = views[:, i]
                view_j = views[:, j]
                F_ij = fundamental_matrices[:, i, j]

                # Photometric consistency
                photo_loss = self.photometric_loss(view_i, view_j)
                total_loss += self.weight_photo * photo_loss

                # Epipolar consistency (if correspondences available)
                if correspondences is not None:
                    pts_i = correspondences[:, i]
                    pts_j = correspondences[:, j]
                    epi_loss = self.epipolar_loss(pts_i, pts_j, F_ij)
                    total_loss += self.weight_epipolar * epi_loss

        return total_loss / (N * (N - 1) / 2)
