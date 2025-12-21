"""
CO3D (Common Objects in 3D) Dataset Preparation Script
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import multiprocessing as mp
from functools import partial
import pickle
import gzip

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CO3DPreparer:
    """Prepare CO3D dataset for MVDiff training."""
    
    # CO3D categories (subset used in MVDiff paper)
    CATEGORIES = [
        'apple', 'backpack', 'ball', 'banana', 'baseballbat',
        'baseballglove', 'bench', 'bicycle', 'book', 'bottle',
        'bowl', 'broccoli', 'cake', 'car', 'carrot',
        'cellphone', 'chair', 'couch', 'cup', 'donut',
        'frisbee', 'hairdryer', 'handbag', 'hydrant', 'keyboard',
        'kite', 'laptop', 'microwave', 'motorcycle', 'mouse',
        'orange', 'parkingmeter', 'pizza', 'plant', 'remote',
        'sandwich', 'skateboard', 'stopsign', 'suitcase', 'teddybear',
        'toaster', 'toilet', 'toybus', 'toyplane', 'toytrain',
        'toytruck', 'tv', 'umbrella', 'vase', 'wineglass'
    ]
    
    # CO3D parameters
    MIN_VIEWS = 8
    MAX_VIEWS = 100
    TARGET_VIEWS = 24  # Number of views to sample
    IMAGE_SIZE = 256
    
    def __init__(
        self,
        co3d_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 4,
        subset_size: Optional[int] = None
    ):
        self.co3d_dir = Path(co3d_dir)
        self.output_dir = Path(output_dir)
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.subset_size = subset_size
        
        # Verify directory exists
        if not self.co3d_dir.exists():
            raise FileNotFoundError(f"CO3D directory not found: {self.co3d_dir}")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
    
    def prepare(self):
        """Main preparation pipeline."""
        logger.info("=" * 60)
        logger.info("CO3D Dataset Preparation")
        logger.info("=" * 60)
        
        # Step 1: Collect sequences
        logger.info("Step 1: Collecting sequences...")
        all_sequences = self._collect_sequences()
        logger.info(f"  Found {len(all_sequences)} valid sequences")
        
        # Step 2: Apply subset if requested
        if self.subset_size and self.subset_size < len(all_sequences):
            logger.info(f"  Selecting subset of {self.subset_size} sequences...")
            np.random.seed(42)
            np.random.shuffle(all_sequences)
            all_sequences = all_sequences[:self.subset_size]
        
        # Step 3: Split dataset
        logger.info("Step 2: Splitting dataset...")
        splits = self._split_dataset(all_sequences)
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} sequences")
        
        # Step 4: Process each split
        for split_name, split_data in splits.items():
            logger.info(f"Step 3: Processing {split_name} split...")
            self._process_split(split_name, split_data)
        
        # Step 5: Create metadata files
        logger.info("Step 4: Creating metadata files...")
        self._create_metadata_files()
        
        # Step 6: Generate statistics
        logger.info("Step 5: Generating statistics...")
        self._generate_statistics()
        
        logger.info("=" * 60)
        logger.info(" Dataset preparation complete!")
        logger.info(f" Output directory: {self.output_dir}")
    
    def _collect_sequences(self) -> List[Dict]:
        """Collect all valid sequences from CO3D."""
        all_sequences = []
        
        for category in tqdm(self.CATEGORIES, desc="Categories"):
            category_dir = self.co3d_dir / category
            
            if not category_dir.exists():
                logger.warning(f"Category {category} not found")
                continue
            
            # Load sequence annotations
            annotation_file = category_dir / "frame_annotations.jgz"
            if annotation_file.exists():
                annotations = self._load_annotations(annotation_file)
            else:
                # Try alternative annotation format
                annotation_file = category_dir / "set_lists" / "set_lists_train.json"
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                else:
                    logger.warning(f"No annotations found for {category}")
                    continue
            
            # Get sequences
            sequences = self._extract_sequences(annotations, category)
            
            for seq in tqdm(sequences, desc=f"  {category}", leave=False):
                sequence_dir = category_dir / seq['sequence_name']
                
                if not sequence_dir.exists():
                    continue
                
                # Count valid frames
                images_dir = sequence_dir / "images"
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.jpg")) + \
                                 list(images_dir.glob("*.png"))
                    
                    if len(image_files) >= self.MIN_VIEWS:
                        sample = {
                            'category': category,
                            'sequence_name': seq['sequence_name'],
                            'sequence_path': str(sequence_dir),
                            'num_frames': len(image_files),
                            'frame_paths': [str(f) for f in sorted(image_files)]
                        }
                        
                        # Add camera data if available
                        cameras_file = sequence_dir / "cameras.json"
                        if cameras_file.exists():
                            sample['has_cameras'] = True
                        
                        # Add point cloud if available
                        pointcloud_file = sequence_dir / "pointcloud.ply"
                        if pointcloud_file.exists():
                            sample['has_pointcloud'] = True
                        
                        all_sequences.append(sample)
        
        return all_sequences
    
    def _load_annotations(self, annotation_file: Path) -> Dict:
        """Load CO3D annotations from compressed JSON."""
        try:
            if annotation_file.suffix == '.jgz':
                with gzip.open(annotation_file, 'rt') as f:
                    return json.load(f)
            else:
                with open(annotation_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading annotations from {annotation_file}: {e}")
            return {}
    
    def _extract_sequences(self, annotations: Dict, category: str) -> List[Dict]:
        """Extract sequence information from annotations."""
        sequences = []
        
        if isinstance(annotations, list):
            # Frame-based annotations
            sequence_names = set()
            for frame in annotations:
                if 'sequence_name' in frame:
                    sequence_names.add(frame['sequence_name'])
            
            sequences = [{'sequence_name': name} for name in sequence_names]
        
        elif isinstance(annotations, dict):
            # Set-based annotations
            if 'train' in annotations:
                sequences.extend([{'sequence_name': s} for s in annotations['train']])
            if 'val' in annotations:
                sequences.extend([{'sequence_name': s} for s in annotations['val']])
        
        return sequences
    
    def _split_dataset(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """Split sequences into train/val/test sets."""
        # Group by category for balanced splitting
        category_sequences = {}
        for seq in sequences:
            cat = seq['category']
            if cat not in category_sequences:
                category_sequences[cat] = []
            category_sequences[cat].append(seq)
        
        # Split each category
        splits = {'train': [], 'val': [], 'test': []}
        
        for cat, cat_seqs in category_sequences.items():
            np.random.seed(42)
            np.random.shuffle(cat_seqs)
            
            n = len(cat_seqs)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            splits['train'].extend(cat_seqs[:n_train])
            splits['val'].extend(cat_seqs[n_train:n_train + n_val])
            splits['test'].extend(cat_seqs[n_train + n_val:])
        
        return splits
    
    def _process_split(self, split_name: str, split_data: List[Dict]):
        """Process a data split."""
        split_dir = self.output_dir / split_name
        metadata = {'samples': []}
        
        # Process sequences
        for seq in tqdm(split_data, desc=f"  Processing {split_name}"):
            result = self._process_sequence(seq, split_dir)
            if result is not None:
                metadata['samples'].append(result)
        
        # Save metadata
        metadata_file = self.output_dir / f"{split_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved {len(metadata['samples'])} sequences to {metadata_file}")
    
    def _process_sequence(self, sequence: Dict, split_dir: Path) -> Optional[Dict]:
        """Process a single sequence."""
        try:
            # Create output directory
            seq_name = f"{sequence['category']}_{sequence['sequence_name']}"
            seq_dir = split_dir / seq_name
            seq_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample frames uniformly
            frame_paths = sequence['frame_paths']
            if len(frame_paths) > self.TARGET_VIEWS:
                # Sample uniformly
                indices = np.linspace(0, len(frame_paths) - 1, self.TARGET_VIEWS, dtype=int)
                sampled_frames = [frame_paths[i] for i in indices]
            else:
                sampled_frames = frame_paths
            
            # Process images
            processed_images = []
            for idx, frame_path in enumerate(sampled_frames):
                src_img_path = Path(frame_path)
                
                if src_img_path.exists():
                    # Load and process image
                    img = Image.open(src_img_path).convert('RGB')
                    
                    # Resize and center crop
                    img = self._center_crop_resize(img, self.IMAGE_SIZE)
                    
                    # Save processed image
                    dst_img_path = seq_dir / f"view_{idx:03d}.png"
                    img.save(dst_img_path, 'PNG')
                    
                    processed_images.append(str(dst_img_path.relative_to(split_dir)))
            
            # Process camera parameters if available
            cameras_file = Path(sequence['sequence_path']) / "cameras.json"
            if cameras_file.exists():
                cameras = self._process_cameras(cameras_file, indices if len(frame_paths) > self.TARGET_VIEWS else None)
                
                # Save camera parameters
                camera_file = seq_dir / "cameras.npz"
                np.savez_compressed(
                    camera_file,
                    intrinsics=cameras['intrinsics'],
                    extrinsics=cameras['extrinsics'],
                    poses=cameras['poses']
                )
            else:
                # Generate default cameras
                cameras = self._generate_default_cameras(len(processed_images))
                camera_file = seq_dir / "cameras.npz"
                np.savez_compressed(
                    camera_file,
                    intrinsics=cameras['intrinsics'],
                    extrinsics=cameras['extrinsics'],
                    poses=cameras['poses']
                )
            
            # Copy point cloud if available
            if sequence.get('has_pointcloud'):
                src_pc = Path(sequence['sequence_path']) / "pointcloud.ply"
                if src_pc.exists():
                    dst_pc = seq_dir / "pointcloud.ply"
                    shutil.copy2(src_pc, dst_pc)
            
            # Create sample metadata
            sample_metadata = {
                'sequence_id': seq_name,
                'category': sequence['category'],
                'images': processed_images,
                'cameras': str(camera_file.relative_to(split_dir)),
                'num_views': len(processed_images),
                'has_pointcloud': sequence.get('has_pointcloud', False)
            }
            
            return sample_metadata
            
        except Exception as e:
            logger.error(f"Error processing {sequence['sequence_name']}: {e}")
            return None
    
    def _center_crop_resize(self, img: Image.Image, size: int) -> Image.Image:
        """Center crop and resize image to target size."""
        # Get dimensions
        width, height = img.size
        
        # Determine crop size (square)
        crop_size = min(width, height)
        
        # Calculate crop coordinates
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Crop and resize
        img = img.crop((left, top, right, bottom))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        return img
    
    def _process_cameras(self, cameras_file: Path, indices: Optional[List[int]] = None) -> Dict:
        """Process CO3D camera parameters."""
        with open(cameras_file, 'r') as f:
            cameras_data = json.load(f)
        
        # Extract relevant frames
        if indices is not None:
            cameras_data = [cameras_data[i] for i in indices if i < len(cameras_data)]
        
        num_views = len(cameras_data)
        intrinsics = []
        extrinsics = []
        poses = []
        
        for cam in cameras_data:
            # Extract intrinsic matrix
            if 'K' in cam:
                K = np.array(cam['K'], dtype=np.float32).reshape(3, 3)
            else:
                # Generate default intrinsics
                focal = self.IMAGE_SIZE
                K = np.array([
                    [focal, 0, self.IMAGE_SIZE / 2],
                    [0, focal, self.IMAGE_SIZE / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            intrinsics.append(K)
            
            # Extract extrinsic matrix
            if 'R' in cam and 'T' in cam:
                R = np.array(cam['R'], dtype=np.float32).reshape(3, 3)
                T = np.array(cam['T'], dtype=np.float32).reshape(3, 1)
                
                # Build 4x4 extrinsic matrix
                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T.squeeze()
                
                extrinsics.append(extrinsic)
                poses.append(np.linalg.inv(extrinsic))
            else:
                # Generate default pose
                extrinsic = np.eye(4, dtype=np.float32)
                extrinsics.append(extrinsic)
                poses.append(extrinsic)
        
        return {
            'intrinsics': np.stack(intrinsics),
            'extrinsics': np.stack(extrinsics),
            'poses': np.stack(poses)
        }
    
    def _generate_default_cameras(self, num_views: int) -> Dict:
        """Generate default camera parameters for sequences without camera data."""
        # Generate circular camera arrangement
        intrinsics = []
        extrinsics = []
        poses = []
        
        radius = 2.5
        
        for i in range(num_views):
            # Intrinsics
            focal = self.IMAGE_SIZE
            K = np.array([
                [focal, 0, self.IMAGE_SIZE / 2],
                [0, focal, self.IMAGE_SIZE / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics.append(K)
            
            # Extrinsics (circular arrangement)
            angle = 2 * np.pi * i / num_views
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.3  # Slight elevation
            
            cam_pos = np.array([x, y, z])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            R = np.stack([right, up, -forward], axis=1)
            t = -R @ cam_pos
            
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            
            extrinsics.append(extrinsic)
            poses.append(np.linalg.inv(extrinsic))
        
        return {
            'intrinsics': np.stack(intrinsics),
            'extrinsics': np.stack(extrinsics),
            'poses': np.stack(poses)
        }
    
    def _create_metadata_files(self):
        """Create additional metadata files."""
        # Create category list
        categories_file = self.output_dir / "categories.json"
        with open(categories_file, 'w') as f:
            json.dump(self.CATEGORIES, f, indent=2)
        
        # Create dataset info
        dataset_info = {
            'dataset': 'CO3D',
            'version': 'v2',
            'categories': self.CATEGORIES,
            'num_categories': len(self.CATEGORIES),
            'image_size': self.IMAGE_SIZE,
            'target_views': self.TARGET_VIEWS,
            'preparation_date': datetime.now().isoformat(),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def _generate_statistics(self):
        """Generate dataset statistics."""
        stats = {
            'splits': {},
            'categories': {},
            'total_sequences': 0,
            'total_images': 0
        }
        
        for split in ['train', 'val', 'test']:
            metadata_file = self.output_dir / f"{split}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                samples = metadata['samples']
                total_images = sum(s['num_views'] for s in samples)
                
                stats['splits'][split] = {
                    'num_sequences': len(samples),
                    'num_images': total_images
                }
                
                stats['total_sequences'] += len(samples)
                stats['total_images'] += total_images
                
                # Count per category
                for sample in samples:
                    cat = sample['category']
                    if cat not in stats['categories']:
                        stats['categories'][cat] = {'train': 0, 'val': 0, 'test': 0}
                    stats['categories'][cat][split] += 1
        
        # Save statistics
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info("-" * 40)
        logger.info(f"Total sequences: {stats['total_sequences']:,}")
        logger.info(f"Total images: {stats['total_images']:,}")
        logger.info("\nPer split:")
        for split, split_stats in stats['splits'].items():
            logger.info(f"  {split:5}: {split_stats['num_sequences']:6,} sequences, "
                       f"{split_stats['num_images']:8,} images")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CO3D dataset for MVDiff training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_co3d.py \\
    --co3d_dir /path/to/co3d_v2 \\
    --output_dir data/co3d_mvdiff
        """
    )
    
    parser.add_argument(
        '--co3d_dir', type=str, required=True,
        help='Path to CO3D dataset directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/co3d_mvdiff',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--subset_size', type=int, default=None,
        help='Process only a subset of sequences (for testing)'
    )
    parser.add_argument(
        '--categories', type=str, nargs='*', default=None,
        help='Specific categories to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        parser.error("Train, val, and test ratios must sum to 1.0")
    
    # Initialize preparer
    preparer = CO3DPreparer(
        co3d_dir=args.co3d_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        subset_size=args.subset_size
    )
    
    # Filter categories if specified
    if args.categories:
        preparer.CATEGORIES = [c for c in preparer.CATEGORIES if c in args.categories]
        logger.info(f"Processing categories: {preparer.CATEGORIES}")
    
    # Run preparation
    preparer.prepare()


if __name__ == "__main__":
    main()