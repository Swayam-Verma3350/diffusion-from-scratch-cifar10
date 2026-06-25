"""
model.py
--------
We use diffusers' UNet2DModel as the backbone architecture (this is the standard
U-Net used by DDPM/Improved-DDPM), and load it from the pretrained checkpoint
"google/ddpm-cifar10-32" -- a DDPM trained on CIFAR-10 at 32x32, exactly matching
our assigned dataset.

Per the assignment requirement to "re-implement or wrap the model architecture
within your own PyTorch codebase", we:
  - wrap the pretrained backbone in our own DiffusionModel class,
  - write our own noise scheduler / sampling math from scratch (schedulers.py),
  - write our own training and generation loops (train.py, sample.py)
instead of just calling a ready-made diffusers pipeline.
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from schedulers import get_scheduler


PRETRAINED_CKPT = "google/ddpm-cifar10-32"


class DiffusionModel(nn.Module):
    """
    Wraps a U-Net noise-predictor together with a NoiseScheduler.

    forward(x0) -> training loss for one batch (random timestep per sample)
    sample(...) -> generate images by running the reverse process
    """

    def __init__(self, schedule: str = "linear", num_timesteps: int = 1000,
                 pretrained: bool = True, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps

        if pretrained:
            # Loads pretrained weights from the Hugging Face Hub.
            self.unet = UNet2DModel.from_pretrained(PRETRAINED_CKPT)
        else:
            # Fallback: a small from-scratch U-Net for quick local testing
            # without internet access (used only by our unit tests / sanity checks).
            self.unet = UNet2DModel(
                sample_size=32,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(64, 128, 128),
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            )

        self.unet.to(device)
        self.scheduler = get_scheduler(schedule, num_timesteps=num_timesteps, device=device)

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """diffusers UNet2DModel returns an object with a `.sample` attribute."""
        return self.unet(x_t, t).sample

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Standard DDPM training objective (Ho et al. 2020, simplified loss):
        sample a random t per image, add noise, ask the U-Net to predict that
        noise, and minimize MSE between predicted and true noise.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        x_t, true_noise = self.scheduler.q_sample(x0, t)
        pred_noise = self.predict_noise(x_t, t)
        loss = nn.functional.mse_loss(pred_noise, true_noise)
        return loss

    @torch.no_grad()
    def sample_ddpm(self, num_images: int, image_size: int = 32, channels: int = 3):
        """Full T-step ancestral sampling (slow, highest fidelity baseline)."""
        x_t = torch.randn(num_images, channels, image_size, image_size, device=self.device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_images,), t, device=self.device, dtype=torch.long)
            noise_pred = self.predict_noise(x_t, t_batch)
            x_t = self.scheduler.ddpm_step(noise_pred, t, x_t)
        return x_t.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, num_images: int, num_inference_steps: int = 50,
                     image_size: int = 32, channels: int = 3, eta: float = 0.0):
        """Fast sampling: skip most of the T timesteps (this is our 'inference
        speed' ablation -- compare quality vs. num_inference_steps)."""
        x_t = torch.randn(num_images, channels, image_size, image_size, device=self.device)
        # evenly spaced subset of the full timestep range, e.g. 1000 -> 50 steps
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_inference_steps).long()
        step_indices = list(reversed(step_indices.tolist()))

        for i, t in enumerate(step_indices):
            t_prev = step_indices[i + 1] if i + 1 < len(step_indices) else -1
            t_batch = torch.full((num_images,), t, device=self.device, dtype=torch.long)
            noise_pred = self.predict_noise(x_t, t_batch)
            x_t = self.scheduler.ddim_step(noise_pred, t, t_prev, x_t, eta=eta)
        return x_t.clamp(-1, 1)
