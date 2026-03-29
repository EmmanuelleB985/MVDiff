"""
Demo for MVDiff
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generate import ViewGenerator
from inference.reconstruct import Reconstructor
from models import MVDiff


def create_sample_image(name="object", size=256):
    """Create a synthetic sample image"""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    if name == "chair":
        # Draw chair
        draw.rectangle(
            [size // 3, size // 4, size * 2 // 3, size * 3 // 4],
            fill="brown",
            outline="black",
            width=2,
        )
        draw.rectangle(
            [size // 3, size // 2, size * 2 // 3, size * 2 // 3],
            fill="brown",
            outline="black",
            width=2,
        )
        for x in [size // 3 + 10, size * 2 // 3 - 10]:
            draw.line([x, size // 2, x, size * 5 // 6], fill="black", width=3)
    elif name == "car":
        # Draw car
        draw.rectangle(
            [size // 4, size // 2, size * 3 // 4, size * 2 // 3],
            fill="red",
            outline="black",
            width=2,
        )
        draw.polygon(
            [
                (size // 3, size // 2),
                (size * 2 // 3, size // 2),
                (size * 3 // 5, size // 3),
                (size * 2 // 5, size // 3),
            ],
            fill="red",
            outline="black",
        )
        for x in [size // 3, size * 2 // 3 - 20]:
            draw.ellipse(
                [x - 15, size * 2 // 3 - 10, x + 15, size * 2 // 3 + 20], fill="black"
            )
    else:
        # Draw generic shape
        draw.ellipse(
            [size // 4, size // 4, size * 3 // 4, size * 3 // 4],
            fill="blue",
            outline="black",
            width=2,
        )

    return img


def setup_model():
    """Setup or load model"""
    checkpoint_path = Path("checkpoints") / "demo_model.pth"

    if not checkpoint_path.exists():
        print("Creating demo model...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        model = MVDiff(
            img_size=256,
            srt_config={"embed_dim": 384, "depth": 6},
            unet_config={"base_channels": 32, "channel_mult": (1, 2, 4)},
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "img_size": 256,
                    "model": {
                        "srt_config": {"embed_dim": 384, "depth": 6},
                        "unet_config": {"base_channels": 32, "channel_mult": (1, 2, 4)},
                        "num_diffusion_steps": 100,
                    },
                },
            },
            checkpoint_path,
        )

    return checkpoint_path


def run_demo():
    """Run complete MVDiff demo"""

    print(" " * 20 + "MVDiff Demo")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create sample images
    print("\n Creating sample images...")
    samples = {
        "chair": create_sample_image("chair"),
        "car": create_sample_image("car"),
        "sphere": create_sample_image("sphere"),
    }

    # Setup model
    checkpoint_path = setup_model()

    # Initialize generator
    print("\n Initializing model...")
    generator = ViewGenerator(checkpoint_path, device=device)

    # Process each sample
    results = {}
    for name, image in samples.items():
        print(f"\n Processing {name}...")

        # Generate views
        views = generator.generate_views(image, num_views=8, num_steps=20)

        # Store results
        results[name] = {"input": image, "views": views}

        # Save outputs
        output_dir = Path("outputs") / f"demo_{name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save input
        image.save(output_dir / "input.png")

        # Save views
        for i, view in enumerate(views):
            Image.fromarray(view).save(output_dir / f"view_{i:02d}.png")

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(results)

    print("\n Demo complete!")
    print("\n Results saved to 'outputs/' directory")
    print("=" * 60)


def create_visualization(results):
    """Create comparison visualization"""
    n_samples = len(results)
    n_views = 4  # Show first 4 views

    fig, axes = plt.subplots(n_samples, n_views + 1, figsize=(15, 3 * n_samples))

    for idx, (name, data) in enumerate(results.items()):
        # Input
        axes[idx, 0].imshow(data["input"])
        axes[idx, 0].set_title(f"{name.title()} - Input", fontweight="bold")
        axes[idx, 0].axis("off")

        # Generated views
        for v in range(min(n_views, len(data["views"]))):
            axes[idx, v + 1].imshow(data["views"][v])
            axes[idx, v + 1].set_title(f"View {v+1}")
            axes[idx, v + 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/demo_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved visualization to outputs/demo_comparison.png")


if __name__ == "__main__":
    run_demo()
