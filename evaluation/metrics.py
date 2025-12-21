"""
Evaluation metrics for multi-view generation
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predicted and target images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Denormalize to [0, 1]
    pred_np = (pred_np + 1) / 2
    target_np = (target_np + 1) / 2
    
    psnr_values = []
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            p = pred_np[i, j].transpose(1, 2, 0)
            t = target_np[i, j].transpose(1, 2, 0)
            psnr_val = psnr_func(t, p, data_range=1.0)
            psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between predicted and target images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Denormalize to [0, 1]
    pred_np = (pred_np + 1) / 2
    target_np = (target_np + 1) / 2
    
    ssim_values = []
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            p = pred_np[i, j].transpose(1, 2, 0)
            t = target_np[i, j].transpose(1, 2, 0)
            ssim_val = ssim_func(t, p, data_range=1.0, channel_axis=2)
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    """Compute LPIPS perceptual distance"""
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        # Return dummy value if lpips not installed
        return 0.0
    
    B, N, C, H, W = pred.shape
    
    # Reshape for batch processing
    pred = pred.reshape(B * N, C, H, W)
    target = target.reshape(B * N, C, H, W)
    
    with torch.no_grad():
        lpips_val = lpips_fn(pred, target)
    
    return lpips_val.mean().item()


def compute_fid(features_real: np.ndarray, features_fake: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (simplified version)
    Would need InceptionV3 features for real implementation
    """
    from scipy import linalg
    
    # Calculate statistics
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    # Calculate Fréchet distance
    diff = mu_real - mu_fake
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)


class MetricTracker:
    """Track metrics during training/evaluation"""
    
    def __init__(self, metrics=['psnr', 'ssim', 'lpips']):
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.values = {metric: [] for metric in self.metrics}
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key in self.values:
                self.values[key].append(value)
    
    def get_average(self) -> dict:
        """Get average of all metrics"""
        averages = {}
        for metric, values in self.values.items():
            if values:
                averages[metric] = np.mean(values)
            else:
                averages[metric] = 0.0
        return averages
    
    def get_std(self) -> dict:
        """Get standard deviation of all metrics"""
        stds = {}
        for metric, values in self.values.items():
            if values:
                stds[f"{metric}_std"] = np.std(values)
            else:
                stds[f"{metric}_std"] = 0.0
        return stds
