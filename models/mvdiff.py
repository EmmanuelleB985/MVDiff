"""
Main MVDiff model combining all components
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attention import compute_fundamental_matrix
from .srt import SceneRepresentationTransformer
from .unet import ViewConditionedUNet


class MVDiff(nn.Module):
    """
    MVDiff: Multi-View Diffusion for 3D Object Reconstruction

    Combines Scene Representation Transformer (SRT) with view-conditioned
    diffusion model for generating consistent multi-view images.
    """

    def __init__(
        self,
        img_size: int = 256,
        img_channels: int = 3,
        srt_config: Dict = None,
        unet_config: Dict = None,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()

        # Default configs
        if srt_config is None:
            srt_config = {"embed_dim": 768, "depth": 12, "num_heads": 12}
        if unet_config is None:
            unet_config = {"base_channels": 64, "channel_mult": (1, 2, 4, 8)}

        # Scene Representation Transformer
        self.srt = SceneRepresentationTransformer(img_size=img_size, **srt_config)

        # View-conditioned U-Net
        self.unet = ViewConditionedUNet(
            img_channels=img_channels,
            scene_embed_dim=srt_config.get("embed_dim", 768),
            **unet_config,
        )

        # Diffusion parameters
        self.num_steps = num_diffusion_steps
        self._setup_diffusion(beta_schedule)

    def _setup_diffusion(self, schedule_type: str):
        """Setup diffusion schedule parameters"""
        if schedule_type == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_steps)
        elif schedule_type == "cosine":
            steps = torch.arange(self.num_steps + 1) / self.num_steps
            alphas = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        )

    def forward(
        self,
        images: torch.Tensor,
        poses: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        fundamental_matrices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            images: [B, N, C, H, W] - multi-view images
            poses: [B, N, 4, 4] - camera poses
            timesteps: [B] - diffusion timesteps
            fundamental_matrices: [B, N, N, 3, 3] - fundamental matrices

        Returns:
            loss: scalar loss tensor
        """
        B, N = images.shape[:2]

        # Get scene representation
        scene_repr = self.srt(images, poses)

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(0, self.num_steps, (B * N,), device=images.device)

        # Add noise and predict
        noise = torch.randn_like(images.flatten(0, 1))
        noisy_images = self.q_sample(images.flatten(0, 1), timesteps, noise)

        # Expand scene representation
        scene_repr_expanded = scene_repr.unsqueeze(1).repeat(1, N, 1, 1)
        scene_repr_expanded = scene_repr_expanded.flatten(0, 1)

        # Predict noise
        noise_pred = self.unet(
            noisy_images, timesteps, scene_repr_expanded, fundamental_matrices
        )

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean image (forward diffusion)"""
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[
            :, None, None, None
        ]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def generate_views(
        self,
        input_images: torch.Tensor,
        input_poses: torch.Tensor,
        target_poses: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate novel views from input images"""
        B, N_out = target_poses.shape[:2]

        # Get scene representation
        scene_repr = self.srt(input_images, input_poses)

        # Initialize with noise
        shape = (
            B,
            N_out,
            self.unet.img_channels,
            input_images.shape[-2],
            input_images.shape[-1],
        )
        x_t = torch.randn(shape, device=input_images.device)

        # Sampling loop
        timesteps = torch.linspace(self.num_steps - 1, 0, num_inference_steps).long()

        for t in timesteps:
            t_batch = torch.full((B * N_out,), t, device=x_t.device)

            # Flatten batch and views
            x_t_flat = x_t.flatten(0, 1)
            scene_repr_expanded = (
                scene_repr.unsqueeze(1).repeat(1, N_out, 1, 1).flatten(0, 1)
            )

            # Predict noise
            noise_pred = self.unet(x_t_flat, t_batch, scene_repr_expanded)

            # Denoise step
            x_t_flat = self.p_sample(x_t_flat, noise_pred, t)
            x_t = x_t_flat.view(B, N_out, *x_t_flat.shape[1:])

        return x_t

    def p_sample(
        self, x_t: torch.Tensor, noise_pred: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Single denoising step"""
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]

        # Compute x_0 prediction
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
            alpha_cumprod_t
        )
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # Compute mean
        mean = (
            torch.sqrt(alpha_cumprod_prev)
            * (1 - alpha_t)
            / (1 - alpha_cumprod_t)
            * x_0_pred
            + torch.sqrt(alpha_t)
            * (1 - alpha_cumprod_prev)
            / (1 - alpha_cumprod_t)
            * x_t
        )

        # Add noise for non-final steps
        if t > 0:
            variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_t)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
