#!/usr/bin/env python3
"""
GSO (Google Scanned Objects) Dataset Preparation Script
Prepares GSO dataset for MVDiff training
"""

import argparse
import io
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GSOPreparer:
    """Prepare Google Scanned Objects dataset for MVDiff training."""

    # GSO parameters
    NUM_VIEWS = 24
    IMAGE_SIZE = 256
    CAMERA_DISTANCE = 2.5

    def __init__(
        self,
        gso_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        render_views: bool = True,
    ):
        self.gso_dir = Path(gso_dir)
        self.output_dir = Path(output_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.render_views = render_views

        # Verify directory exists
        if not self.gso_dir.exists():
            raise FileNotFoundError(f"GSO directory not found: {self.gso_dir}")

        # Create output directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """Main preparation pipeline."""
        logger.info("=" * 60)
        logger.info("GSO Dataset Preparation")
        logger.info("=" * 60)

        # Step 1: Collect objects
        logger.info("Step 1: Collecting objects...")
        all_objects = self._collect_objects()
        logger.info(f"  Found {len(all_objects)} valid objects")

        # Step 2: Split dataset
        logger.info("Step 2: Splitting dataset...")
        splits = self._split_dataset(all_objects)
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} objects")

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

    def _collect_objects(self) -> List[Dict]:
        """Collect all valid GSO objects."""
        all_objects = []

        # GSO objects are organized by ID
        object_dirs = [d for d in self.gso_dir.iterdir() if d.is_dir()]

        for obj_dir in tqdm(object_dirs, desc="Collecting objects"):
            # Check for required files
            mesh_file = obj_dir / "meshes" / "model.obj"
            texture_file = obj_dir / "materials" / "textures" / "texture.png"

            if mesh_file.exists():
                obj_info = {
                    "object_id": obj_dir.name,
                    "object_path": str(obj_dir),
                    "mesh_file": str(mesh_file),
                    "has_texture": texture_file.exists(),
                }

                # Check for metadata
                metadata_file = obj_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    obj_info["metadata"] = metadata
                    obj_info["category"] = metadata.get("category", "unknown")
                else:
                    obj_info["category"] = "unknown"

                all_objects.append(obj_info)

        return all_objects

    def _split_dataset(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """Split objects into train/val/test sets."""
        # Group by category for balanced splitting
        category_objects = {}
        for obj in objects:
            cat = obj["category"]
            if cat not in category_objects:
                category_objects[cat] = []
            category_objects[cat].append(obj)

        # Split each category
        splits = {"train": [], "val": [], "test": []}

        for cat, cat_objs in category_objects.items():
            np.random.seed(42)
            np.random.shuffle(cat_objs)

            n = len(cat_objs)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            splits["train"].extend(cat_objs[:n_train])
            splits["val"].extend(cat_objs[n_train : n_train + n_val])
            splits["test"].extend(cat_objs[n_train + n_val :])

        return splits

    def _process_split(self, split_name: str, split_data: List[Dict]):
        """Process a data split."""
        split_dir = self.output_dir / split_name
        metadata = {"samples": []}

        for obj in tqdm(split_data, desc=f"  Processing {split_name}"):
            result = self._process_object(obj, split_dir)
            if result is not None:
                metadata["samples"].append(result)

        # Save metadata
        metadata_file = self.output_dir / f"{split_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Saved {len(metadata['samples'])} objects to {metadata_file}")

    def _process_object(self, obj_info: Dict, split_dir: Path) -> Optional[Dict]:
        """Process a single GSO object."""
        try:
            # Create output directory
            obj_name = f"{obj_info['category']}_{obj_info['object_id']}"
            obj_dir = split_dir / obj_name
            obj_dir.mkdir(parents=True, exist_ok=True)

            if self.render_views:
                # Render multi-view images from 3D model
                images = self._render_views(
                    obj_info["mesh_file"], obj_info["has_texture"]
                )
            else:
                # Look for existing rendered images
                images = self._load_existing_views(Path(obj_info["object_path"]))

            # Save rendered images
            image_paths = []
            for idx, img in enumerate(images):
                img_path = obj_dir / f"view_{idx:03d}.png"
                img.save(img_path, "PNG")
                image_paths.append(str(img_path.relative_to(split_dir)))

            # Generate camera parameters
            cameras = self._generate_camera_parameters()

            # Save camera parameters
            camera_file = obj_dir / "cameras.npz"
            np.savez_compressed(
                camera_file,
                intrinsics=cameras["intrinsics"],
                extrinsics=cameras["extrinsics"],
                poses=cameras["poses"],
            )

            # Copy mesh file
            src_mesh = Path(obj_info["mesh_file"])
            dst_mesh = obj_dir / "mesh.obj"
            shutil.copy2(src_mesh, dst_mesh)

            # Copy texture if available
            if obj_info["has_texture"]:
                src_texture = (
                    Path(obj_info["object_path"])
                    / "materials"
                    / "textures"
                    / "texture.png"
                )
                if src_texture.exists():
                    dst_texture = obj_dir / "texture.png"
                    shutil.copy2(src_texture, dst_texture)

            # Create sample metadata
            sample_metadata = {
                "object_id": obj_name,
                "category": obj_info["category"],
                "gso_id": obj_info["object_id"],
                "images": image_paths,
                "cameras": str(camera_file.relative_to(split_dir)),
                "mesh": str(dst_mesh.relative_to(split_dir)),
                "num_views": len(image_paths),
                "has_texture": obj_info["has_texture"],
            }

            # Add additional metadata if available
            if "metadata" in obj_info:
                sample_metadata["metadata"] = obj_info["metadata"]

            return sample_metadata

        except Exception as e:
            logger.error(f"Error processing {obj_info['object_id']}: {e}")
            return None

    def _render_views(self, mesh_file: str, has_texture: bool) -> List[Image.Image]:
        """Render multi-view images from 3D mesh."""
        try:
            # Load mesh
            mesh = trimesh.load(mesh_file, force="mesh")

            # Normalize mesh
            mesh.vertices -= mesh.vertices.mean(axis=0)
            max_extent = np.max(np.abs(mesh.vertices))
            mesh.vertices /= max_extent

            # Create scene
            scene = trimesh.Scene([mesh])

            # Render views
            images = []

            for i in range(self.NUM_VIEWS):
                # Calculate camera position
                angle = 2 * np.pi * i / self.NUM_VIEWS
                elevation = np.radians(15) if i % 2 == 0 else np.radians(-15)

                x = self.CAMERA_DISTANCE * np.cos(elevation) * np.cos(angle)
                y = self.CAMERA_DISTANCE * np.sin(elevation)
                z = self.CAMERA_DISTANCE * np.cos(elevation) * np.sin(angle)

                camera_pose = self._look_at(
                    eye=[x, y, z], target=[0, 0, 0], up=[0, 1, 0]
                )

                # Render image
                try:
                    # Set up camera
                    scene.camera.resolution = [self.IMAGE_SIZE, self.IMAGE_SIZE]
                    scene.camera.fov = [60, 60]
                    scene.camera_transform = camera_pose

                    # Render
                    image_data = scene.save_image(
                        resolution=[self.IMAGE_SIZE, self.IMAGE_SIZE]
                    )
                    img = Image.open(io.BytesIO(image_data))

                except Exception:
                    # Fallback: create blank image
                    img = Image.new("RGB", (self.IMAGE_SIZE, self.IMAGE_SIZE), "white")

                images.append(img)

            return images

        except Exception as e:
            logger.warning(f"Could not render mesh {mesh_file}: {e}")
            # Return blank images as fallback
            return [
                Image.new("RGB", (self.IMAGE_SIZE, self.IMAGE_SIZE), "white")
                for _ in range(self.NUM_VIEWS)
            ]

    def _load_existing_views(self, object_path: Path) -> List[Image.Image]:
        """Load existing rendered views if available."""
        images = []

        # Check for rendered images directory
        renders_dir = object_path / "renders"
        if renders_dir.exists():
            image_files = sorted(renders_dir.glob("*.png"))[: self.NUM_VIEWS]

            for img_file in image_files:
                img = Image.open(img_file).convert("RGB")
                img = img.resize(
                    (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.LANCZOS
                )
                images.append(img)

        # Fill remaining with blank images
        while len(images) < self.NUM_VIEWS:
            images.append(Image.new("RGB", (self.IMAGE_SIZE, self.IMAGE_SIZE), "white"))

        return images

    def _look_at(self, eye, target, up):
        """Create look-at matrix for camera."""
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        matrix = np.eye(4)
        matrix[:3, 0] = right
        matrix[:3, 1] = up
        matrix[:3, 2] = -forward
        matrix[:3, 3] = eye

        return matrix

    def _generate_camera_parameters(self) -> Dict[str, np.ndarray]:
        """Generate camera parameters for GSO objects."""
        intrinsics = []
        extrinsics = []
        poses = []

        focal = self.IMAGE_SIZE

        for i in range(self.NUM_VIEWS):
            # Intrinsic matrix
            K = np.array(
                [
                    [focal, 0, self.IMAGE_SIZE / 2],
                    [0, focal, self.IMAGE_SIZE / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            intrinsics.append(K)

            # Calculate camera position
            angle = 2 * np.pi * i / self.NUM_VIEWS
            elevation = np.radians(15) if i % 2 == 0 else np.radians(-15)

            x = self.CAMERA_DISTANCE * np.cos(elevation) * np.cos(angle)
            y = self.CAMERA_DISTANCE * np.sin(elevation)
            z = self.CAMERA_DISTANCE * np.cos(elevation) * np.sin(angle)

            cam_pos = np.array([x, y, z])

            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            # Rotation matrix
            R = np.stack([right, up, -forward], axis=1)
            t = -R @ cam_pos

            # 4x4 extrinsic matrix
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            extrinsics.append(extrinsic)
            poses.append(np.linalg.inv(extrinsic))

        return {
            "intrinsics": np.stack(intrinsics),
            "extrinsics": np.stack(extrinsics),
            "poses": np.stack(poses),
        }

    def _create_metadata_files(self):
        """Create additional metadata files."""
        # Create dataset info
        dataset_info = {
            "dataset": "GSO",
            "version": "1.0",
            "image_size": self.IMAGE_SIZE,
            "num_views_per_object": self.NUM_VIEWS,
            "camera_distance": self.CAMERA_DISTANCE,
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
        stats = {"splits": {}, "categories": {}, "total_objects": 0, "total_images": 0}

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


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Google Scanned Objects dataset for MVDiff training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_gso.py \\
    --gso_dir /path/to/gso-dataset \\
    --output_dir data/gso_mvdiff
        """,
    )

    parser.add_argument(
        "--gso_dir", type=str, required=True, help="Path to GSO dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/gso_mvdiff",
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
        "--render_views",
        action="store_true",
        default=True,
        help="Render views from 3D models (default: True)",
    )
    parser.add_argument(
        "--no_render",
        action="store_false",
        dest="render_views",
        help="Use existing rendered views if available",
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        parser.error("Train, val, and test ratios must sum to 1.0")

    # Initialize preparer
    preparer = GSOPreparer(
        gso_dir=args.gso_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        render_views=args.render_views,
    )

    # Run preparation
    preparer.prepare()


if __name__ == "__main__":
    main()
