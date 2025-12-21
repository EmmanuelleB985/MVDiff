"""
Multi-view generation from single images
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from typing import List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MVDiff


class ViewGenerator:
    """Generate multiple views from a single image"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.denormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
    
    def _load_model(self, checkpoint_path: str) -> MVDiff:
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Extract model config
        model_config = config.get('model', {})
        
        model = MVDiff(
            img_size=model_config.get('img_size', 256),
            srt_config=model_config.get('srt_config', {}),
            unet_config=model_config.get('unet_config', {}),
            num_diffusion_steps=model_config.get('num_diffusion_steps', 100)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def generate_camera_poses(
        self,
        num_views: int,
        radius: float = 2.5,
        elevation: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera poses in circular arrangement"""
        poses = []
        
        for i in range(num_views):
            azimuth = 2 * np.pi * i / num_views
            
            # Camera position
            x = radius * np.cos(azimuth)
            y = radius * elevation
            z = radius * np.sin(azimuth)
            cam_pos = np.array([x, y, z])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Rotation matrix
            R = np.stack([right, up, -forward], axis=1)
            
            # Translation
            t = -R @ cam_pos
            
            # 4x4 pose matrix
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            poses.append(pose)
        
        poses = torch.from_numpy(np.stack(poses))
        
        # Generate intrinsics
        focal = 256 / (2 * np.tan(np.radians(30)))
        K = torch.tensor([
            [focal, 0, 128],
            [0, focal, 128],
            [0, 0, 1]
        ], dtype=torch.float32)
        intrinsics = K.unsqueeze(0).repeat(num_views, 1, 1)
        
        return poses, intrinsics
    
    @torch.no_grad()
    def generate_views(
        self,
        input_image: Image.Image,
        num_views: int = 16,
        num_steps: int = 50
    ) -> List[np.ndarray]:
        """Generate multiple views from single image"""
        
        # Prepare input
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        # Generate camera poses
        target_poses, _ = self.generate_camera_poses(num_views)
        target_poses = target_poses.unsqueeze(0).to(self.device)
        
        # Use identity pose for input
        input_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).to(self.device)
        input_images = input_tensor.unsqueeze(1)
        
        # Generate views
        generated_views = self.model.generate_views(
            input_images,
            input_pose,
            target_poses,
            num_inference_steps=num_steps
        )
        
        # Convert to numpy images
        views = []
        for i in range(num_views):
            view = generated_views[0, i]
            view = self.denormalize(view)
            view = torch.clamp(view, 0, 1)
            view = view.permute(1, 2, 0).cpu().numpy()
            view = (view * 255).astype(np.uint8)
            views.append(view)
        
        return views
    
    def generate_video(
        self,
        input_image: Image.Image,
        output_path: str,
        num_frames: int = 60,
        fps: int = 30
    ):
        """Generate rotating video from single image"""
        import cv2
        
        # Generate views
        views = self.generate_views(input_image, num_views=num_frames)
        
        # Create video writer
        height, width = views[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for view in views:
            frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
