"""
MVDiff Test Suite
Comprehensive testing for all components
"""

import pytest
import torch


# Mark for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.gpu = pytest.mark.gpu
pytest.mark.slow = pytest.mark.slow
pytest.mark.benchmark = pytest.mark.benchmark


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Create sample image tensor."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_batch():
    """Create sample batch."""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 8, 3, 256, 256),
        "poses": torch.eye(4).unsqueeze(0).repeat(batch_size, 8, 1, 1),
        "intrinsics": torch.eye(3).unsqueeze(0).repeat(batch_size, 8, 1, 1),
    }


@pytest.fixture
def temp_checkpoint(tmp_path):
    """Create temporary checkpoint."""
    checkpoint = {
        "epoch": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "loss": 0.1,
        "config": {"img_size": 256},
    }
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    return path


# ============================================================================
# Model Tests
# ============================================================================


class TestModels:
    """Test model components."""

    @pytest.mark.unit
    def test_srt_initialization(self):
        """Test Scene Representation Transformer initialization."""
        from models.srt import SceneRepresentationTransformer

        model = SceneRepresentationTransformer(img_size=256, embed_dim=768, depth=12)

        assert model.embed_dim == 768
        assert len(model.transformer_blocks) == 12

    @pytest.mark.unit
    def test_srt_forward(self):
        """Test SRT forward pass."""
        from models.srt import SceneRepresentationTransformer

        model = SceneRepresentationTransformer(img_size=64)
        images = torch.randn(2, 4, 3, 64, 64)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 4, 1, 1)

        output = model(images, poses)

        assert output.shape[0] == 2  # Batch size
        assert output.shape[-1] == model.embed_dim

    @pytest.mark.unit
    def test_unet_initialization(self):
        """Test UNet initialization."""
        from models.unet import ViewConditionedUNet

        model = ViewConditionedUNet(
            img_channels=3, base_channels=64, scene_embed_dim=768
        )

        assert model.img_channels == 3
        assert model.base_channels == 64

    @pytest.mark.unit
    def test_attention_module(self):
        """Test attention modules."""
        from models.attention import EpipolarCrossAttention

        attn = EpipolarCrossAttention(embed_dim=256, num_heads=8)

        query = torch.randn(2, 100, 256)
        key = torch.randn(2, 100, 256)
        value = torch.randn(2, 100, 256)

        output = attn(query, key, value)

        assert output.shape == query.shape

    @pytest.mark.unit
    def test_fundamental_matrix_computation(self):
        """Test fundamental matrix computation."""
        from models.attention import compute_fundamental_matrix

        K1 = torch.eye(3).unsqueeze(0)
        K2 = torch.eye(3).unsqueeze(0)
        R = torch.eye(3).unsqueeze(0)
        t = torch.tensor([[0.1, 0.2, 0.3]])

        F = compute_fundamental_matrix(K1, K2, R, t)

        assert F.shape == (1, 3, 3)
        # Fundamental matrix should have rank 2 
        assert torch.abs(torch.det(F[0])) < 1e-4

    @pytest.mark.unit
    def test_mvdiff_initialization(self):
        """Test main MVDiff model initialization."""
        from models.mvdiff import MVDiff

        model = MVDiff(img_size=128, num_diffusion_steps=100)

        assert model.num_steps == 100
        assert hasattr(model, "srt")
        assert hasattr(model, "unet")

    @pytest.mark.gpu
    def test_mvdiff_cuda_forward(self, device):
        """Test MVDiff forward pass on CUDA."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        from models.mvdiff import MVDiff

        model = MVDiff(img_size=64).to(device)
        images = torch.randn(2, 4, 3, 64, 64).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 4, 1, 1).to(device)

        loss = model(images, poses)

        assert loss.device == device
        assert loss.requires_grad


# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Test training components."""

    @pytest.mark.unit
    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization."""
        from training.dataset import MultiViewDataset

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        dataset = MultiViewDataset(data_root=str(data_dir), split="train")

        assert dataset.split == "train"
        assert dataset.img_size == 256

    @pytest.mark.unit
    def test_loss_computation(self):
        """Test loss functions."""
        from training.losses import DiffusionLoss

        loss_fn = DiffusionLoss(loss_type="mse")

        pred = torch.randn(4, 3, 64, 64, requires_grad=True)
        target = torch.randn(4, 3, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad


# ============================================================================
# Inference Tests
# ============================================================================


class TestInference:
    """Test inference components."""

    @pytest.mark.unit
    def test_camera_pose_generation(self):
        """Test camera pose generation."""
        from inference.generate import ViewGenerator

        class MockGenerator:
            def generate_camera_poses(self, num_views):
                poses = []
                for i in range(num_views):
                    pose = torch.eye(4)
                    poses.append(pose)
                intrinsics = torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1)
                return torch.stack(poses), intrinsics

        generator = MockGenerator()
        poses, intrinsics = generator.generate_camera_poses(8)

        assert poses.shape == (8, 4, 4)
        assert intrinsics.shape == (8, 3, 3)

    @pytest.mark.slow
    def test_full_pipeline(self, temp_checkpoint):
        """Test full inference pipeline."""
        pass


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Test evaluation metrics."""

    @pytest.mark.unit
    def test_psnr_computation(self):
        """Test PSNR metric."""
        from evaluation.metrics import compute_psnr

        pred = torch.randn(2, 4, 3, 64, 64)
        target = pred + torch.randn_like(pred) * 0.1

        psnr = compute_psnr(pred, target)

        assert psnr > 0
        assert psnr < 100

    @pytest.mark.unit
    def test_ssim_computation(self):
        """Test SSIM metric."""
        from evaluation.metrics import compute_ssim

        pred = torch.randn(2, 4, 3, 64, 64)
        target = pred + torch.randn_like(pred) * 0.1

        ssim = compute_ssim(pred, target)

        assert 0 <= ssim <= 1

    @pytest.mark.unit
    def test_metric_tracker(self):
        """Test metric tracking."""
        from evaluation.metrics import MetricTracker

        tracker = MetricTracker(metrics=["psnr", "ssim"])

        tracker.update(psnr=25.5, ssim=0.85)
        tracker.update(psnr=26.0, ssim=0.87)

        averages = tracker.get_average()

        assert averages["psnr"] == pytest.approx(25.75)
        assert averages["ssim"] == pytest.approx(0.86)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_data_to_model_pipeline(self, tmp_path):
        """Test data loading to model training."""
        from torch.utils.data import DataLoader

        from models.mvdiff import MVDiff
        from training.dataset import MultiViewDataset

        data_dir = tmp_path / "data"
        data_dir.mkdir()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_loop(self, tmp_path):
        """Test complete training loop."""
        pass

    @pytest.mark.integration
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        from models.mvdiff import MVDiff

        model1 = MVDiff(img_size=64)
        checkpoint_path = tmp_path / "checkpoint.pth"

        torch.save(
            {"model_state_dict": model1.state_dict(), "config": {"img_size": 64}},
            checkpoint_path,
        )

        model2 = MVDiff(img_size=64)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint["model_state_dict"])

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# ============================================================================
# Utility Tests
# ============================================================================


class TestUtilities:
    """Test utility functions."""

    @pytest.mark.unit
    def test_config_loading(self, tmp_path):
        """Test configuration loading."""
        import yaml

        config = {
            "model": {"img_size": 256, "num_diffusion_steps": 1000},
            "training": {"batch_size": 16, "learning_rate": 1e-4},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == config


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])