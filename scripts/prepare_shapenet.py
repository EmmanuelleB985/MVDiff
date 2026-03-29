"""
ShapeNet Dataset Preparation Script
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ShapeNetPreparer:
    """Prepare ShapeNet dataset for MVDiff training."""

    # ShapeNet categories used in MVDiff paper
    CATEGORIES = {
        "02691156": "airplane",
        "02828884": "bench",
        "02933112": "cabinet",
        "02958343": "car",
        "03001627": "chair",
        "03211117": "display",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "rifle",
        "04256520": "sofa",
        "04379243": "table",
        "04401088": "telephone",
        "04530566": "watercraft",
    }

    # Standard ShapeNet rendering parameters
    NUM_VIEWS = 24
    IMAGE_SIZE = 256
    FOCAL_LENGTH = 131.25  # Fixed focal length used in ShapeNet
    CAMERA_DISTANCE = 2.732

    def __init__(
        self,
        shapenet_dir: str,
        rendering_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 4,
    ):
        self.shapenet_dir = Path(shapenet_dir)
        self.rendering_dir = Path(rendering_dir)
        self.output_dir = Path(output_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers

        # Verify directories exist
        if not self.shapenet_dir.exists():
            raise FileNotFoundError(
                f"ShapeNet directory not found: {self.shapenet_dir}"
            )
        if not self.rendering_dir.exists():
            raise FileNotFoundError(
                f"Rendering directory not found: {self.rendering_dir}"
            )

        # Create output directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """Main preparation pipeline."""
        logger.info("=" * 60)
        logger.info("ShapeNet Dataset Preparation")
        logger.info("=" * 60)

        # Step 1: Collect all samples
        logger.info("Step 1: Collecting samples...")
        all_samples = self._collect_samples()
        logger.info(f"  Found {len(all_samples)} valid samples")

        # Step 2: Split dataset
        logger.info("Step 2: Splitting dataset...")
        splits = self._split_dataset(all_samples)
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")

        # Step 3: Process each split
        for split_name, split_data in splits.items():
            logger.info(f"Step 3: Processing {split_name} split...")
            self._process_split(split_name, split_data)

        # Step 4: Create metadata files
        logger.info("Step 4: Creating metadata files...")
        self._create_metadata_files()

        # Step 5: Generate statistics
        logger.info("Step 5: Generating statistics...")
        self._generate_statistics()

        logger.info("=" * 60)
        logger.info(" Dataset preparation complete!")
        logger.info(f" Output directory: {self.output_dir}")

    def _collect_samples(self) -> List[Dict]:
        """Collect all valid samples from ShapeNet."""
        all_samples = []

        for cat_id, cat_name in tqdm(self.CATEGORIES.items(), desc="Categories"):
            cat_rendering_dir = self.rendering_dir / cat_id

            if not cat_rendering_dir.exists():
                logger.warning(
                    f"Category {cat_name} ({cat_id}) not found in renderings"
                )
                continue

            # Get all object IDs
            object_ids = [d.name for d in cat_rendering_dir.iterdir() if d.is_dir()]

            for obj_id in tqdm(object_ids, desc=f"  {cat_name}", leave=False):
                obj_rendering_dir = cat_rendering_dir / obj_id / "rendering"

                if not obj_rendering_dir.exists():
                    continue

                # Check if all views are present
                images = list(obj_rendering_dir.glob("*.png"))

                if len(images) >= self.NUM_VIEWS:
                    sample = {
                        "category_id": cat_id,
                        "category": cat_name,
                        "object_id": obj_id,
                        "rendering_path": str(obj_rendering_dir),
                        "num_views": len(images),
                    }

                    # Check if 3D model exists
                    model_path = (
                        self.shapenet_dir
                        / cat_id
                        / obj_id
                        / "models"
                        / "model_normalized.obj"
                    )
                    if model_path.exists():
                        sample["model_path"] = str(model_path)

                    all_samples.append(sample)

        return all_samples

    def _split_dataset(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Split samples into train/val/test sets."""
        # Shuffle samples
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(samples)

        n_samples = len(samples)
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train : n_train + n_val],
            "test": samples[n_train + n_val :],
        }

        return splits

    def _process_split(self, split_name: str, split_data: List[Dict]):
        """Process a data split."""
        split_dir = self.output_dir / split_name
        metadata = {"samples": []}

        # Use multiprocessing for faster processing
        with mp.Pool(self.num_workers) as pool:
            process_func = partial(self._process_sample, split_dir=split_dir)
            results = list(
                tqdm(
                    pool.imap(process_func, split_data),
                    total=len(split_data),
                    desc=f"  Processing {split_name}",
                )
            )

        # Filter out None results and add to metadata
        metadata["samples"] = [r for r in results if r is not None]

        # Save metadata
        metadata_file = self.output_dir / f"{split_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Saved {len(metadata['samples'])} samples to {metadata_file}")

    def _process_sample(self, sample: Dict, split_dir: Path) -> Dict:
        """Process a single sample."""
        try:
            # Create output directory for this object
            obj_name = f"{sample['category']}_{sample['object_id']}"
            obj_dir = split_dir / obj_name
            obj_dir.mkdir(parents=True, exist_ok=True)

            # Copy and preprocess images
            rendering_dir = Path(sample["rendering_path"])
            image_paths = []

            for view_idx in range(self.NUM_VIEWS):
                src_img_path = rendering_dir / f"{view_idx:02d}.png"

                if src_img_path.exists():
                    # Load and process image
                    img = Image.open(src_img_path).convert("RGB")

                    # Resize if needed
                    if img.size != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                        img = img.resize(
                            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.LANCZOS
                        )

                    # Save processed image
                    dst_img_path = obj_dir / f"view_{view_idx:03d}.png"
                    img.save(dst_img_path, "PNG")

                    image_paths.append(str(dst_img_path.relative_to(split_dir)))

            # Generate camera parameters
            cameras = self._generate_camera_parameters()

            # Save camera parameters
            camera_file = obj_dir / "cameras.npz"
            np.savez_compressed(
                camera_file,
                intrinsics=cameras["intrinsics"],
                extrinsics=cameras["extrinsics"],
                poses=cameras["poses"],
                focal_length=self.FOCAL_LENGTH,
                image_size=self.IMAGE_SIZE,
            )

            # Copy 3D model if available
            if "model_path" in sample:
                src_model = Path(sample["model_path"])
                dst_model = obj_dir / "model.obj"
                shutil.copy2(src_model, dst_model)

            # Create sample metadata
            sample_metadata = {
                "object_id": obj_name,
                "category": sample["category"],
                "category_id": sample["category_id"],
                "images": image_paths,
                "cameras": str(camera_file.relative_to(split_dir)),
                "num_views": len(image_paths),
                "has_3d_model": "model_path" in sample,
            }

            return sample_metadata

        except Exception as e:
            logger.error(f"Error processing {sample['object_id']}: {e}")
            return None

    def _generate_camera_parameters(self) -> Dict[str, np.ndarray]:
        """Generate ShapeNet camera parameters."""
        num_views = self.NUM_VIEWS

        # Intrinsic matrix (same for all views)
        K = np.array(
            [
                [self.FOCAL_LENGTH, 0, self.IMAGE_SIZE / 2],
                [0, self.FOCAL_LENGTH, self.IMAGE_SIZE / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsics = np.tile(K[np.newaxis], (num_views, 1, 1))

        # Generate extrinsic matrices
        extrinsics = []
        poses = []

        # ShapeNet uses 2 elevations and 12 azimuths
        view_idx = 0
        for elev_idx, elevation in enumerate([30, -30]):  # Two elevations in degrees
            for azim_idx in range(12):  # 12 azimuth angles
                azimuth = azim_idx * 30  # Every 30 degrees

                # Convert to radians
                elev_rad = np.radians(elevation)
                azim_rad = np.radians(azimuth)

                # Camera position in spherical coordinates
                x = self.CAMERA_DISTANCE * np.cos(elev_rad) * np.sin(azim_rad)
                y = self.CAMERA_DISTANCE * np.sin(elev_rad)
                z = self.CAMERA_DISTANCE * np.cos(elev_rad) * np.cos(azim_rad)
                cam_pos = np.array([x, y, z])

                # Look at origin
                forward = -cam_pos / np.linalg.norm(cam_pos)
                right = np.cross(np.array([0, 1, 0]), forward)
                right = right / np.linalg.norm(right)
                up = np.cross(forward, right)

                # Rotation matrix (world to camera)
                R = np.stack([right, -up, forward], axis=0)  # Note: -up for y-axis flip

                # Translation
                t = -R @ cam_pos

                # 4x4 extrinsic matrix
                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t

                extrinsics.append(extrinsic)
                poses.append(np.linalg.inv(extrinsic))  # Camera to world

                view_idx += 1

        extrinsics = np.stack(extrinsics)
        poses = np.stack(poses)

        return {"intrinsics": intrinsics, "extrinsics": extrinsics, "poses": poses}

    def _create_metadata_files(self):
        """Create additional metadata files."""
        # Create category mapping
        category_file = self.output_dir / "categories.json"
        with open(category_file, "w") as f:
            json.dump(self.CATEGORIES, f, indent=2)

        # Create camera configuration
        camera_config = {
            "num_views": self.NUM_VIEWS,
            "image_size": self.IMAGE_SIZE,
            "focal_length": self.FOCAL_LENGTH,
            "camera_distance": self.CAMERA_DISTANCE,
            "elevations": [30, -30],
            "azimuths": list(range(0, 360, 30)),
        }

        camera_config_file = self.output_dir / "camera_config.json"
        with open(camera_config_file, "w") as f:
            json.dump(camera_config, f, indent=2)

        # Create dataset info
        dataset_info = {
            "dataset": "ShapeNet",
            "version": "v2",
            "categories": list(self.CATEGORIES.values()),
            "num_categories": len(self.CATEGORIES),
            "image_size": self.IMAGE_SIZE,
            "num_views_per_object": self.NUM_VIEWS,
            "preparation_date": datetime.now().isoformat(),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }

        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

    def _generate_statistics(self):
        """Generate dataset statistics."""
        stats = {"splits": {}, "categories": {}, "total_images": 0, "total_objects": 0}

        for split in ["train", "val", "test"]:
            metadata_file = self.output_dir / f"{split}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                samples = metadata["samples"]
                stats["splits"][split] = {
                    "num_objects": len(samples),
                    "num_images": len(samples) * self.NUM_VIEWS,
                }

                stats["total_objects"] += len(samples)
                stats["total_images"] += len(samples) * self.NUM_VIEWS

                # Count per category
                for sample in samples:
                    cat = sample["category"]
                    if cat not in stats["categories"]:
                        stats["categories"][cat] = {"train": 0, "val": 0, "test": 0}
                    stats["categories"][cat][split] += 1

        # Save statistics
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info("-" * 40)
        logger.info(f"Total objects: {stats['total_objects']:,}")
        logger.info(f"Total images: {stats['total_images']:,}")
        logger.info("\nPer split:")
        for split, split_stats in stats["splits"].items():
            logger.info(
                f"  {split:5}: {split_stats['num_objects']:6,} objects, "
                f"{split_stats['num_images']:8,} images"
            )
        logger.info("\nPer category:")
        for cat, cat_stats in stats["categories"].items():
            total = sum(cat_stats.values())
            logger.info(
                f"  {cat:12}: {total:5,} "
                f"(train: {cat_stats['train']}, "
                f"val: {cat_stats['val']}, "
                f"test: {cat_stats['test']})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ShapeNet dataset for MVDiff training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_shapenet.py \\
    --shapenet_dir /path/to/ShapeNetCore.v2 \\
    --rendering_dir /path/to/ShapeNetRendering \\
    --output_dir data/shapenet_mvdiff
        """,
    )

    parser.add_argument(
        "--shapenet_dir",
        type=str,
        required=True,
        help="Path to ShapeNetCore.v2 directory",
    )
    parser.add_argument(
        "--rendering_dir",
        type=str,
        required=True,
        help="Path to ShapeNetRendering directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/shapenet_mvdiff",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Specific categories to process (default: all)",
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        parser.error("Train, val, and test ratios must sum to 1.0")

    # Initialize preparer
    preparer = ShapeNetPreparer(
        shapenet_dir=args.shapenet_dir,
        rendering_dir=args.rendering_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )

    # Filter categories if specified
    if args.categories:
        preparer.CATEGORIES = {
            k: v for k, v in preparer.CATEGORIES.items() if v in args.categories
        }
        logger.info(f"Processing categories: {list(preparer.CATEGORIES.values())}")

    # Run preparation
    preparer.prepare()


if __name__ == "__main__":
    main()
