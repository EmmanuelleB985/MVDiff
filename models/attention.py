"""
Attention modules with epipolar geometry constraints
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EpipolarCrossAttention(nn.Module):
    """
    Cross-attention with epipolar geometry constraints.
    Enforces geometric consistency between different views.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def compute_epipolar_mask(
        self, fundamental_matrix: torch.Tensor, h: int, w: int, threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Compute epipolar line constraints for attention masking.

        Args:
            fundamental_matrix: [B, 3, 3] - fundamental matrix between views
            h, w: height and width of feature maps
            threshold: distance threshold for epipolar constraint
        Returns:
            mask: [B, h*w, h*w] - attention mask based on epipolar geometry
        """
        B = fundamental_matrix.shape[0]
        device = fundamental_matrix.device

        # Create grid of pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        coords = torch.stack(
            [x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=-1
        )
        coords = coords.float()

        # Compute epipolar lines: l = F^T * p
        epipolar_lines = torch.einsum(
            "bij,nj->bni", fundamental_matrix.transpose(1, 2), coords
        )

        # Normalize lines
        epipolar_lines = epipolar_lines / (
            epipolar_lines.norm(dim=-1, keepdim=True) + 1e-8
        )

        # Compute point-to-line distances
        distances = torch.abs(torch.einsum("bni,mj->bnm", epipolar_lines, coords))

        # Create mask based on distance threshold
        mask = (distances < threshold).float()

        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fundamental_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, L, D] - input features
            fundamental_matrix: [B, 3, 3] - optional fundamental matrix
        Returns:
            output: [B, L, D] - output with epipolar-constrained attention
        """
        B, L, D = query.shape

        # Linear projections
        Q = self.q_proj(query).reshape(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(key).reshape(B, L, self.num_heads, self.head_dim)
        V = self.v_proj(value).reshape(B, L, self.num_heads, self.head_dim)

        # Transpose for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply epipolar constraint if provided
        if fundamental_matrix is not None:
            h = w = int(math.sqrt(L))
            epipolar_mask = self.compute_epipolar_mask(fundamental_matrix, h, w)
            epipolar_mask = epipolar_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(epipolar_mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(attn_out)

        return output


class MultiViewAttentionBlock(nn.Module):
    """Self-attention block for multi-view consistency"""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.epipolar_attn = EpipolarCrossAttention(channels, num_heads)

    def forward(
        self, x: torch.Tensor, fundamental_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Reshape for attention
        x_norm = self.norm(x)
        x_flat = rearrange(x_norm, "b c h w -> b (h w) c")

        # Apply epipolar-constrained attention
        attn_out = self.epipolar_attn(x_flat, x_flat, x_flat, fundamental_matrix)

        # Reshape back
        attn_out = rearrange(attn_out, "b (h w) c -> b c h w", h=H, w=W)

        return x + attn_out


def compute_fundamental_matrix(
    K1: torch.Tensor, K2: torch.Tensor, R: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Compute fundamental matrix from camera parameters.

    Args:
        K1, K2: [B, 3, 3] - intrinsic matrices
        R: [B, 3, 3] - rotation matrix from cam1 to cam2
        t: [B, 3] - translation vector from cam1 to cam2
    Returns:
        F: [B, 3, 3] - fundamental matrix
    """
    # Compute essential matrix
    t_cross = torch.zeros_like(R)
    t_cross[:, 0, 1] = -t[:, 2]
    t_cross[:, 0, 2] = t[:, 1]
    t_cross[:, 1, 0] = t[:, 2]
    t_cross[:, 1, 2] = -t[:, 0]
    t_cross[:, 2, 0] = -t[:, 1]
    t_cross[:, 2, 1] = t[:, 0]

    E = torch.matmul(t_cross, R)

    # Compute fundamental matrix: F = K2^(-T) * E * K1^(-1)
    K1_inv = torch.inverse(K1)
    K2_inv_t = torch.inverse(K2).transpose(1, 2)

    F = torch.matmul(torch.matmul(K2_inv_t, E), K1_inv)

    # Normalize
    F = F / (F[:, -1, -1:, None] + 1e-8)

    return F
