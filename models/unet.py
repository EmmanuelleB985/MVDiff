"""
View-Conditioned U-Net for diffusion model
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import MultiViewAttentionBlock


class ViewConditionedUNet(nn.Module):
    """
    U-Net with view conditioning and multi-view attention.
    Incorporates epipolar geometry constraints for consistent generation.
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16,),
        dropout: float = 0.1,
        scene_embed_dim: int = 768,
        time_embed_dim: int = 256,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.base_channels = base_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Scene conditioning
        self.scene_proj = nn.Linear(scene_embed_dim, time_embed_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(now_channels, out_channels, time_embed_dim, dropout)
                )
                now_channels = out_channels
                channels.append(now_channels)

                # Add attention at specified resolutions
                if 2 ** (len(channel_mult) - level - 1) in attention_resolutions:
                    self.encoder_blocks.append(
                        MultiViewAttentionBlock(now_channels, num_heads=8)
                    )
                    channels.append(now_channels)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(
                    nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1)
                )
                channels.append(now_channels)

        # Middle blocks
        self.middle_blocks = nn.ModuleList(
            [
                ResBlock(now_channels, now_channels, time_embed_dim, dropout),
                MultiViewAttentionBlock(now_channels, num_heads=8),
                ResBlock(now_channels, now_channels, time_embed_dim, dropout),
            ]
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()

        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_channels = base_channels * mult

            for i in range(num_res_blocks + 1):
                self.decoder_blocks.append(
                    ResBlock(
                        now_channels + channels.pop(),
                        out_channels,
                        time_embed_dim,
                        dropout,
                    )
                )
                now_channels = out_channels

                # Add attention at specified resolutions
                if (
                    2 ** (len(channel_mult) - level - 1) in attention_resolutions
                    and i == num_res_blocks
                ):
                    self.decoder_blocks.append(
                        MultiViewAttentionBlock(now_channels, num_heads=8)
                    )

            # Upsample (except last level)
            if level != 0:
                self.decoder_blocks.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, 4, stride=2, padding=1
                    )
                )

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, img_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        scene_repr: torch.Tensor,
        fundamental_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - noisy image
            t: [B] - timestep
            scene_repr: [B, L, D] - scene representation
            fundamental_matrix: [B, 3, 3] - optional fundamental matrix
        Returns:
            output: [B, C, H, W] - predicted noise
        """
        # Time and scene conditioning
        t_emb = self.time_embed(t)
        scene_emb = self.scene_proj(scene_repr.mean(dim=1))  # Global pooling
        cond_emb = t_emb + scene_emb

        # Initial convolution
        h = self.conv_in(x)

        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            if isinstance(block, ResBlock):
                h = block(h, cond_emb)
            elif isinstance(block, MultiViewAttentionBlock):
                h = block(h, fundamental_matrix)
            else:  # Downsampling
                h = block(h)
            skip_connections.append(h)

        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, ResBlock):
                h = block(h, cond_emb)
            elif isinstance(block, MultiViewAttentionBlock):
                h = block(h, fundamental_matrix)

        # Decoder with skip connections
        for block in self.decoder_blocks:
            if isinstance(block, ResBlock):
                if skip_connections:
                    h = torch.cat([h, skip_connections.pop()], dim=1)
                h = block(h, cond_emb)
            elif isinstance(block, MultiViewAttentionBlock):
                h = block(h, fundamental_matrix)
            else:  # Upsampling
                h = block(h)

        # Final convolution
        output = self.conv_out(h)

        return output


class ResBlock(nn.Module):
    """Residual block with time/scene conditioning"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
