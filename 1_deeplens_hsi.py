"""An example for hyperspectral image reconstruction.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import logging
import os
import random
import shutil
import string
from datetime import datetime

# import lpips
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens.network import NAFNet, PerceptualLoss
from deeplens.utils import batch_psnr, batch_ssim, set_logger, set_seed
from deeplens.hsi_camera import HSICamera
from hsi_dataset import CaveDataset


def config():
    """Load and prepare configuration."""
    # Load config files
    with open("configs/1_deeplens_hsi.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Set up result directory
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = f"{current_time}-HSI-Recon-{random_string}"

    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    # Set random seed
    if args["seed"] is None:
        args["seed"] = random.randint(0, 1000)
    set_seed(args["seed"])

    # Configure logging
    set_logger(result_dir)
    logging.info(f"Experiment: {args['exp_name']}")
    if not args["is_debug"]:
        raise Exception("Add your wandb logging config here.")

    # Configure device
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"Using {torch.cuda.get_device_name(0)} GPU")

    # Save config and code
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)
    shutil.copy("1_deeplens_hsi.py", f"{result_dir}/1_deeplens_hsi.py")

    return args


class Trainer:
    """Class for training models."""

    def __init__(self, args):
        """Initialize the trainer.

        Args:
            args: Dictionary with training configuration
        """
        self.args = args
        self.device = args["device"]

        # Initialize model, renderer, and data
        self._init_data(
            train_set_config=args["train_set"], eval_set_config=args["eval_set"]
        )
        self._init_model(net_args=args["network"], train_args=args["train"])
        self._init_camera(camera_args=args["camera"])

    def _init_camera(self, camera_args):
        """Initialize the camera."""
        self.camera = HSICamera(
            lens_file=camera_args["lens_file"],
            sensor_file=camera_args["sensor_file"],
            device=self.device,
        )

    def _init_model(self, net_args, train_args):
        """Initialize the model and optimizer."""
        # Create model
        self.model = NAFNet(
            in_chan=net_args["in_chan"],
            out_chan=net_args["out_chan"],
            width=net_args["width"],
            middle_blk_num=net_args["middle_blk_num"],
            enc_blk_nums=net_args["enc_blk_nums"],
            dec_blk_nums=net_args["dec_blk_nums"],
        )
        self.model.to(self.device)

        # Load checkpoint if provided
        if net_args.get("ckpt_path"):
            state_dict = torch.load(net_args["ckpt_path"], map_location=self.device)
            try:
                self.model.load_state_dict(state_dict["model"])
            except:
                self.model.load_state_dict(state_dict)

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=float(train_args["lr"])
        )

        # Create scheduler
        total_steps = train_args["epochs"] * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-7,
        )

        # Create loss
        self.lpips_loss = PerceptualLoss(device=self.device)

        # Create metrics
        # self.lpips_metric = lpips.LPIPS(net="alex").to(self.device)

    def _init_data(self, train_set_config, eval_set_config):
        """Initialize data loaders."""
        # Create datasets
        train_dataset = CaveDataset(
            file_path=train_set_config["dataset"],
            crop_size=train_set_config["crop_size"],
            is_train=True,
        )
        val_dataset = CaveDataset(
            file_path=eval_set_config["dataset"],
            crop_size=eval_set_config["crop_size"],
            is_train=False,
        )

        # Create data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_set_config["batch_size"],
            shuffle=True,
            num_workers=train_set_config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=eval_set_config["batch_size"],
            shuffle=False,
            num_workers=eval_set_config["num_workers"],
            pin_memory=True,
        )

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        # Save model state
        torch.save(
            self.model.state_dict(),
            f"{self.args['result_dir']}/network_epoch{epoch}.pth",
        )

        # Save optimizer state
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{self.args['result_dir']}/optimizer_epoch{epoch}.pth",
        )

    def compute_loss(self, inputs, targets):
        """Compute loss between model outputs and targets.

        Args:
            inputs: Input blurred images [B, C, H, W]
            targets: Target clean images [B, C, H, W]

        Returns:
            loss: The computed loss value
            loss_dict: Dictionary with loss components for logging
        """
        # Forward pass
        outputs = self.model(inputs)
        outputs = outputs.clamp(0, 1)

        # L1 loss on spectral channels
        loss = F.l1_loss(outputs, targets)

        # Total loss
        loss_dict = {
            "total_loss": loss.item(),
        }
        return loss, loss_dict, outputs

    def compute_metrics(self, outputs, targets=None):
        """Compute metrics between model outputs and targets."""
        # Calculate metrics
        l1_score = F.l1_loss(outputs, targets)
        ssim_score = batch_ssim(outputs, targets)
        psnr_score = batch_psnr(outputs, targets)

        metrics = {
            "l1": l1_score,
            "ssim": ssim_score,
            "psnr": psnr_score,
        }
        return metrics

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        # Training loop
        for i, data_dict in enumerate(tqdm(self.train_loader)):
            # Image simulation
            # inputs: [B, 3, H, W]
            # targets: [B, 31, H, W]
            inputs, targets = self.camera.render(data_dict)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Compute loss
            loss, loss_dict, outputs = self.compute_loss(inputs, targets)
            total_loss += loss_dict["total_loss"]

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Log progress
            if (
                i % self.args["train"]["log_every_n_steps"]
                == self.args["train"]["log_every_n_steps"] - 1
            ):
                print(
                    f"Epoch: {epoch + 1}/{self.args['train']['epochs']}, "
                    f"Batch: {i + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss_dict['total_loss']:.4f}"
                )

                # Save sample images
                with torch.no_grad():
                    inputs_rgb = inputs.clone()
                    outputs_rgb = self.camera.spectral2rgb(outputs)
                    targets_rgb = self.camera.spectral2rgb(targets)
                    save_image(
                        torch.cat([inputs_rgb, outputs_rgb, targets_rgb], dim=2),
                        f"{self.args['result_dir']}/train_epoch{epoch}_batch{i}.png",
                    )

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch + 1}/{self.args['train']['epochs']} completed.")
        print(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch):
        """Run validation."""
        # Set model to eval mode
        self.model.eval()

        # Initialize metrics
        val_psnr = 0.0
        val_ssim = 0.0
        val_samples = 0

        # Validation loop
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Apply blur to create inputs using camera
                # inputs: [B, 3, H, W]
                # targets: [B, 31, H, W]
                inputs, targets = self.camera.render(data_dict)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                outputs = outputs.clamp(0, 1)

                # Compute metrics
                metrics = self.compute_metrics(outputs, targets)
                val_psnr += metrics["psnr"] * inputs.size(0)
                val_ssim += metrics["ssim"] * inputs.size(0)
                val_samples += inputs.size(0)

                # Save sample validation images
                if i == 0:
                    # Convert to RGB
                    inputs_rgb = inputs.clone()
                    outputs_rgb = self.camera.spectral2rgb(outputs)
                    targets_rgb = self.camera.spectral2rgb(targets)

                    # Save images
                    save_image(
                        torch.cat([inputs_rgb, outputs_rgb, targets_rgb], dim=2),
                        f"{self.args['result_dir']}/val_epoch{epoch}_{i}.png",
                    )

        # Calculate average metrics
        metrics = {}
        if val_samples > 0:
            metrics["val_psnr"] = val_psnr / val_samples
            metrics["val_ssim"] = val_ssim / val_samples

        # Log epoch results
        print("-" * 50)
        print(f"Validation PSNR: {metrics.get('val_psnr', 'N/A')} dB")
        print(f"Validation SSIM: {metrics.get('val_ssim', 'N/A')}")
        print("-" * 50)

        return metrics

    def train(self):
        """Run the full training process."""
        best_psnr = 0.0

        for epoch in range(self.args["train"]["epochs"]):
            # Train one epoch
            self.train_epoch(epoch)

            # Validate and save checkpoint
            if (epoch + 1) % self.args["train"]["eval_every_n_epochs"] == 0:
                metrics = self.validate_epoch(epoch)

                # Save checkpoint
                self.save_checkpoint(epoch + 1)

                # Save best model
                if metrics.get("val_psnr", 0) > best_psnr:
                    best_psnr = metrics.get("val_psnr", 0)
                    torch.save(
                        self.model.state_dict(),
                        f"{self.args['result_dir']}/best_model.pth",
                    )
                    print(f"New best model saved with PSNR: {best_psnr:.4f}")

        # Save final model
        self.save_checkpoint(self.args["train"]["epochs"])
        print("Training completed!")


def main():
    """Main function to start the single GPU training."""
    # Training configuration
    args = config()

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
