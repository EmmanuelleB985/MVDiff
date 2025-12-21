"""
Scene Representation Transformer (SRT) for learning implicit 3D representations
"""

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SceneRepresentationTransformer(nn.Module):
    """
    SRT for learning implicit 3D representations from input views.
    Encodes spatial and view information into unified latent representations.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        camera_embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Camera pose encoding
        self.camera_embed = nn.Sequential(
            nn.Linear(16, camera_embed_dim),  # 4x4 matrix flattened
            nn.ReLU(),
            nn.Linear(camera_embed_dim, camera_embed_dim),
            nn.ReLU(),
            nn.Linear(camera_embed_dim, embed_dim)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, images: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, N, C, H, W] - N input views
            camera_poses: [B, N, 4, 4] - camera matrices
        Returns:
            scene_repr: [B, L, D] - scene representation tokens
        """
        B, N, C, H, W = images.shape
        
        # Extract patches
        images = rearrange(images, 'b n c h w -> (b n) c h w')
        patches = self.patch_embed(images)
        patches = rearrange(patches, 'bn d h w -> bn (h w) d')
        
        # Encode camera poses
        camera_poses = rearrange(camera_poses, 'b n h w -> b n (h w)')
        camera_embeds = self.camera_embed(camera_poses)
        camera_embeds = repeat(camera_embeds, 'b n d -> b n p d', p=self.num_patches)
        camera_embeds = rearrange(camera_embeds, 'b n p d -> b (n p) d')
        
        # Combine patches and camera info
        patches = rearrange(patches, '(b n) p d -> b (n p) d', b=B)
        x = patches + camera_embeds
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final projection
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
