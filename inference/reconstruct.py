"""
3D reconstruction from multi-view images
"""

import numpy as np
import torch
from typing import List, Optional
import cv2


class Reconstructor:
    """3D reconstruction from generated multi-view images"""
    
    def __init__(self):
        self.method = 'mvs'  # Multi-view stereo by default
    
    def reconstruct_mvs(
        self,
        images: List[np.ndarray],
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> dict:
        """
        Multi-view stereo reconstruction.
        Returns point cloud and mesh.
        """
        point_cloud = []
        colors = []
        
        # Simple depth estimation from stereo pairs
        for i in range(len(images) - 1):
            img1, img2 = images[i], images[i + 1]
            pose1, pose2 = poses[i].numpy(), poses[i + 1].numpy()
            K = intrinsics[i].numpy()
            
            # Estimate depth
            depth_map = self._estimate_depth(img1, img2, pose1, pose2, K)
            
            # Backproject to 3D
            points_3d, point_colors = self._backproject(img1, depth_map, K, pose1)
            
            point_cloud.append(points_3d)
            colors.append(point_colors)
        
        # Combine all points
        if point_cloud:
            point_cloud = np.vstack(point_cloud)
            colors = np.vstack(colors)
        else:
            point_cloud = np.zeros((0, 3))
            colors = np.zeros((0, 3))
        
        return {
            'points': point_cloud,
            'colors': colors,
            'method': 'mvs'
        }
    
    def _estimate_depth(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pose1: np.ndarray,
        pose2: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """Estimate depth using stereo matching"""
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Stereo matching
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
        
        # Calculate baseline
        pos1 = np.linalg.inv(pose1)[:3, 3]
        pos2 = np.linalg.inv(pose2)[:3, 3]
        baseline = np.linalg.norm(pos2 - pos1)
        
        # Convert disparity to depth
        focal = K[0, 0]
        disparity[disparity <= 0] = 1
        depth = (baseline * focal) / disparity
        depth = np.clip(depth, 0.1, 10.0)
        
        return depth
    
    def _backproject(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        pose: np.ndarray
    ) -> tuple:
        """Backproject image pixels to 3D points"""
        h, w = depth.shape
        
        # Create pixel grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize pixel coordinates
        xx = (xx - K[0, 2]) / K[0, 0]
        yy = (yy - K[1, 2]) / K[1, 1]
        
        # 3D points in camera space
        points_cam = np.stack([
            xx * depth,
            yy * depth,
            depth
        ], axis=-1)
        
        # Transform to world space
        points_cam_homo = np.concatenate([
            points_cam.reshape(-1, 3),
            np.ones((h * w, 1))
        ], axis=1)
        
        world_transform = np.linalg.inv(pose)
        points_world = (world_transform @ points_cam_homo.T).T[:, :3]
        
        # Get colors
        colors = image.reshape(-1, 3) / 255.0
        
        # Filter invalid points
        valid_mask = depth.reshape(-1) < 9.0
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]
        
        return points_world, colors
    
    def reconstruct_nerf(
        self,
        images: List[np.ndarray],
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        resolution: int = 64
    ) -> dict:
        """
        Simplified NeRF-style volumetric reconstruction.
        Returns voxel grid.
        """
        # Create voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution))
        voxel_colors = np.zeros((resolution, resolution, resolution, 3))
        
        # Define bounding box
        bbox_min, bbox_max = -1.5, 1.5
        
        # Voxel coordinates
        x = np.linspace(bbox_min, bbox_max, resolution)
        
        # Process each view (simplified)
        for view_idx, img in enumerate(images):
            pose_inv = torch.inverse(poses[view_idx]).numpy()
            K = intrinsics[view_idx].numpy()
            
            # Sample voxels (simplified version)
            for i in range(0, resolution, 4):  # Skip some for speed
                for j in range(0, resolution, 4):
                    for k in range(0, resolution, 4):
                        # World point
                        point = np.array([x[i], x[j], x[k], 1])
                        
                        # Transform to camera
                        point_cam = pose_inv @ point
                        
                        if point_cam[2] <= 0:
                            continue
                        
                        # Project to image
                        point_img = K @ point_cam[:3]
                        u = int(point_img[0] / point_img[2])
                        v = int(point_img[1] / point_img[2])
                        
                        # Check bounds
                        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                            voxel_grid[i, j, k] += 1
                            voxel_colors[i, j, k] += img[v, u] / 255.0
        
        # Average colors
        mask = voxel_grid > 0
        voxel_colors[mask] /= voxel_grid[mask, np.newaxis]
        
        # Threshold for occupancy
        voxel_grid = (voxel_grid > np.percentile(voxel_grid[voxel_grid > 0], 50)).astype(float)
        
        return {
            'voxels': voxel_grid,
            'colors': voxel_colors,
            'bbox': (bbox_min, bbox_max),
            'method': 'nerf'
        }
    
    def save_point_cloud(self, points: np.ndarray, colors: np.ndarray, path: str):
        """Save point cloud as PLY file"""
        # Simple PLY writer
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point, color in zip(points, colors):
                color_int = (color * 255).astype(int)
                f.write(f"{point[0]} {point[1]} {point[2]} ")
                f.write(f"{color_int[0]} {color_int[1]} {color_int[2]}\n")
        
        print(f"Point cloud saved to {path}")
