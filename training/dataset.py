"""
Dataset loaders for multi-view training
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """
    Dataset for loading multi-view images with camera parameters.
    Supports ShapeNet, CO3D, and GSO datasets.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 256,
        num_views: int = 24,
        augment: bool = True,
        return_fundamental: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.num_views = num_views
        self.augment = augment and split == "train"
        self.return_fundamental = return_fundamental

        # Load metadata
        self.samples = self._load_metadata()

        # Setup transforms
        self.transform = self._get_transforms()

    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        metadata_path = self.data_root / f"{self.split}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata["samples"]
        else:
            # Scan directory structure
            samples = []
            for obj_dir in self.data_root.glob("*"):
                if obj_dir.is_dir():
                    sample = {
                        "object_id": obj_dir.name,
                        "images": sorted([str(f) for f in obj_dir.glob("*.png")]),
                        "cameras": str(obj_dir / "cameras.npz"),
                    }
                    if len(sample["images"]) >= self.num_views:
                        samples.append(sample)
            return samples

    def _get_transforms(self) -> transforms.Compose:
        """Get image transformations"""
        transform_list = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if self.augment:
            transform_list.insert(1, transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))

        return transforms.Compose(transform_list)

    def _load_camera_params(self, camera_path: str) -> Dict[str, np.ndarray]:
        """Load camera parameters"""
        if Path(camera_path).exists():
            cameras = np.load(camera_path)
            return {
                "intrinsics": cameras["intrinsics"],
                "extrinsics": cameras.get("extrinsics", cameras.get("poses")),
            }
        else:
            # Generate default cameras
            return self._generate_default_cameras()

    def _generate_default_cameras(self) -> Dict[str, np.ndarray]:
        """Generate default camera parameters"""
        num_views = 24
        radius = 2.5

        # Intrinsics
        focal = self.img_size / (2 * np.tan(np.radians(30)))
        K = np.array(
            [[focal, 0, self.img_size / 2], [0, focal, self.img_size / 2], [0, 0, 1]]
        )
        intrinsics = np.tile(K[np.newaxis], (num_views, 1, 1))

        # Extrinsics (circular arrangement)
        extrinsics = []
        for i in range(num_views):
            theta = 2 * np.pi * i / num_views

            # Camera position
            cam_pos = np.array([radius * np.cos(theta), 0.5, radius * np.sin(theta)])

            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            # Rotation matrix
            R = np.stack([right, up, -forward], axis=1)

            # Translation
            t = -R @ cam_pos

            # 4x4 extrinsic matrix
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            extrinsics.append(extrinsic)

        return {"intrinsics": intrinsics, "extrinsics": np.stack(extrinsics)}

    def _compute_fundamental_matrices(
        self, intrinsics: torch.Tensor, poses: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute fundamental matrices between all view pairs"""
        if not self.return_fundamental:
            return None

        from models.attention import compute_fundamental_matrix

        N = poses.shape[0]
        F_matrices = torch.zeros(N, N, 3, 3)

        for i in range(N):
            for j in range(N):
                if i != j:
                    # Relative transformation
                    rel_pose = torch.matmul(torch.inverse(poses[j]), poses[i])
                    R = rel_pose[:3, :3].unsqueeze(0)
                    t = rel_pose[:3, 3].unsqueeze(0)

                    # Compute fundamental matrix
                    F = compute_fundamental_matrix(
                        intrinsics[i : i + 1, :3, :3],
                        intrinsics[j : j + 1, :3, :3],
                        R,
                        t,
                    )
                    F_matrices[i, j] = F[0]

        return F_matrices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load images
        image_paths = sample["images"][: self.num_views]
        images = []

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        images = torch.stack(images)

        # Load camera parameters
        cameras = self._load_camera_params(sample.get("cameras", ""))
        intrinsics = torch.from_numpy(cameras["intrinsics"][: self.num_views]).float()
        poses = torch.from_numpy(cameras["extrinsics"][: self.num_views]).float()

        # Compute fundamental matrices
        fundamental_matrices = self._compute_fundamental_matrices(intrinsics, poses)

        output = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "object_id": sample["object_id"],
        }

        if fundamental_matrices is not None:
            output["fundamental_matrices"] = fundamental_matrices

        return output


class CO3DDataset(MultiViewDataset):
    """Specific loader for CO3D dataset"""

    def _load_metadata(self) -> List[Dict]:
        """Load CO3D specific metadata"""
        # CO3D has a different structure
        # Implementation would handle CO3D's frame-based organization
        return super()._load_metadata()


class GSODataset(MultiViewDataset):
    """Specific loader for Google Scanned Objects"""

    def _load_metadata(self) -> List[Dict]:
        """Load GSO specific metadata"""
        # GSO has high-quality scans with specific structure
        # Implementation would handle GSO's organization
        return super()._load_metadata()
