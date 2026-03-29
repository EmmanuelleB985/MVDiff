"""
Main training script for MVDiff
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MVDiff
from training.dataset import MultiViewDataset
from training.losses import DiffusionLoss


class Trainer:
    """Main trainer class for MVDiff"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model
        self.model = MVDiff(**config["model"]).to(self.device)

        # Setup datasets
        self.train_dataset = MultiViewDataset(
            data_root=config["data"]["root"], split="train", **config["data"]["train"]
        )
        self.val_dataset = MultiViewDataset(
            data_root=config["data"]["root"], split="val", **config["data"]["val"]
        )

        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup loss
        self.criterion = DiffusionLoss()

        # Setup logging
        self.writer = SummaryWriter(config["logging"]["tensorboard_dir"])
        if config["logging"].get("use_wandb", False):
            wandb.init(project=config["logging"]["wandb_project"], config=config)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        opt_config = self.config["optimizer"]

        if opt_config["type"] == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config["lr"],
                betas=(opt_config.get("beta1", 0.9), opt_config.get("beta2", 0.999)),
            )
        elif opt_config["type"] == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        sched_config = self.config.get("scheduler")
        if sched_config is None:
            return None

        if sched_config["type"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_config["T_max"]
            )
        elif sched_config["type"] == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config["step_size"],
                gamma=sched_config["gamma"],
            )
        else:
            return None

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["images"].to(self.device)
            poses = batch["poses"].to(self.device)
            fundamental_matrices = batch.get("fundamental_matrices")
            if fundamental_matrices is not None:
                fundamental_matrices = fundamental_matrices.to(self.device)

            # Forward pass
            loss = self.model(images, poses, fundamental_matrices=fundamental_matrices)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config["training"].get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["gradient_clip"]
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to tensorboard
            if self.global_step % self.config["logging"]["log_interval"] == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            # Move to device
            images = batch["images"].to(self.device)
            poses = batch["poses"].to(self.device)
            fundamental_matrices = batch.get("fundamental_matrices")
            if fundamental_matrices is not None:
                fundamental_matrices = fundamental_matrices.to(self.device)

            # Forward pass
            loss = self.model(images, poses, fundamental_matrices=fundamental_matrices)

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)

        # Log validation metrics
        self.writer.add_scalar("val/loss", avg_loss, self.epoch)

        return avg_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val loss: {self.best_val_loss:.4f}")

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if epoch % self.config["logging"]["val_interval"] == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

            # Regular checkpoint
            if epoch % self.config["logging"]["checkpoint_interval"] == 0:
                self.save_checkpoint()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train MVDiff model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shapenet.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override data root if provided
    config["data"]["root"] = args.data_root

    # Create trainer
    trainer = Trainer(config)

    # Resume if checkpoint provided
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {trainer.epoch}")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
