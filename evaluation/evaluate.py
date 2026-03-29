"""
Evaluation script for MVDiff model
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from models import MVDiff
from training.dataset import MultiViewDataset


class Evaluator:
    """Evaluate MVDiff model on test set"""

    def __init__(self, checkpoint_path: str, data_root: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Load dataset
        self.dataset = MultiViewDataset(
            data_root=data_root, split="test", augment=False
        )

        self.dataloader = DataLoader(
            self.dataset, batch_size=4, shuffle=False, num_workers=4
        )

    def _load_model(self, checkpoint_path: str) -> MVDiff:
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})

        model = MVDiff(**config.get("model", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        return model

    @torch.no_grad()
    def evaluate(self, num_samples: int = None) -> dict:
        """Run evaluation on test set"""
        print("Starting evaluation...")

        all_metrics = {"psnr": [], "ssim": [], "lpips": []}

        num_batches = (
            len(self.dataloader)
            if num_samples is None
            else min(num_samples // 4, len(self.dataloader))
        )

        pbar = tqdm(self.dataloader, total=num_batches, desc="Evaluating")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_batches:
                break

            # Move to device
            images = batch["images"].to(self.device)
            poses = batch["poses"].to(self.device)
            B, N = images.shape[:2]

            # Use first view as input
            input_images = images[:, :1]
            input_poses = poses[:, :1]
            target_images = images[:, 1:]
            target_poses = poses[:, 1:]

            # Generate views
            generated_views = self.model.generate_views(
                input_images, input_poses, target_poses, num_inference_steps=50
            )

            # Compute metrics
            psnr = compute_psnr(generated_views, target_images)
            ssim = compute_ssim(generated_views, target_images)
            lpips = compute_lpips(generated_views, target_images, self.device)

            all_metrics["psnr"].append(psnr)
            all_metrics["ssim"].append(ssim)
            all_metrics["lpips"].append(lpips)

            # Update progress bar
            pbar.set_postfix(
                {
                    "PSNR": np.mean(all_metrics["psnr"]),
                    "SSIM": np.mean(all_metrics["ssim"]),
                    "LPIPS": np.mean(all_metrics["lpips"]),
                }
            )

        # Compute final metrics
        results = {
            "PSNR": float(np.mean(all_metrics["psnr"])),
            "PSNR_std": float(np.std(all_metrics["psnr"])),
            "SSIM": float(np.mean(all_metrics["ssim"])),
            "SSIM_std": float(np.std(all_metrics["ssim"])),
            "LPIPS": float(np.mean(all_metrics["lpips"])),
            "LPIPS_std": float(np.std(all_metrics["lpips"])),
            "num_samples": len(all_metrics["psnr"]) * 4,
        }

        return results

    def save_results(self, results: dict, output_path: str):
        """Save evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:15} : {value:.4f}")
            else:
                print(f"{key:15} : {value}")
        print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MVDiff model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint, data_root=args.data_root, device=args.device
    )

    # Run evaluation
    results = evaluator.evaluate(num_samples=args.num_samples)

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
