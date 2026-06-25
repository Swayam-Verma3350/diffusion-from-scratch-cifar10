"""
schedulers.py
--------------
Implements the forward (noising) and reverse (denoising) diffusion process
from first principles, following Ho et al. 2020 (DDPM) and Song et al. 2020 (DDIM).

We write this ourselves (instead of using diffusers' built-in scheduler classes)
so that:
  1. We understand and can clearly explain the math in the report.
  2. We can easily swap noise schedules (linear vs cosine) for the ablation study.

Only the noise *schedule* and sampling *equations* are custom. The neural network
that predicts noise (the U-Net) is wrapped from a pretrained checkpoint in model.py.
"""

import math
import torch


class NoiseScheduler:
    """
    Base class. Subclasses only need to define `_make_betas`.

    Notation (standard DDPM notation):
        beta_t      : variance added at step t
        alpha_t     = 1 - beta_t
        alpha_bar_t = product of alpha_1 ... alpha_t
    """

    def __init__(self, num_timesteps: int = 1000, device="cpu"):
        self.T = num_timesteps
        self.device = device

        betas = self._make_betas(num_timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        # convenience tensors used repeatedly during sampling
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _make_betas(self, T):
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ---------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Add noise to a clean image x0 to produce x_t, for an arbitrary batch of
        timesteps t (one per sample in the batch).

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omab * noise
        return x_t, noise

    # ---------------------------------------------------------------
    # Reverse process: DDPM ancestral sampling (Ho et al. 2020, Eq. 11)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def ddpm_step(self, model_output: torch.Tensor, t: int, x_t: torch.Tensor):
        """
        Single reverse step x_t -> x_{t-1} using the model's noise prediction.
        `t` is a single int (same timestep applied to the whole batch).

        NOTE: this version predicts x0 and clips it to [-1, 1] before computing
        the posterior mean (the standard "clip_sample" safeguard used by
        diffusers' own DDPMScheduler). Without this, the 1/sqrt(alpha_t)
        multiplier blows up for any schedule with beta_t close to 1 (which the
        cosine schedule has near t=T) -- a small noise-prediction error gets
        amplified every step and the whole trajectory diverges to +-inf over
        ~1000 steps, well before the final clamp(-1, 1) is reached.
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)

        # predict x0 from the current noise estimate, then clip it -- this is
        # what keeps the recursion numerically bounded
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # posterior mean of q(x_{t-1} | x_t, x0) (Ho et al. 2020, Eq. 7)
        mean = (
            (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t) * pred_x0
            + (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t) * x_t
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            # posterior variance (tighter than using beta_t directly)
            variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean  # no noise added on the last step

        return x_prev

    # ---------------------------------------------------------------
    # Reverse process: DDIM deterministic sampling (Song et al. 2020)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def ddim_step(self, model_output: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor, eta: float = 0.0):
        """
        Single DDIM reverse step from timestep t to an earlier timestep t_prev.
        Allows skipping steps (t_prev < t - 1), which is what makes DDIM fast.

        eta=0.0  -> fully deterministic DDIM
        eta=1.0  -> recovers DDPM-style stochastic sampling
        """
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

        # predicted x0 from the predicted noise
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * model_output
        noise = torch.randn_like(x_t) if eta > 0 else 0.0

        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
        return x_prev


class LinearScheduler(NoiseScheduler):
    """The original DDPM schedule: betas increase linearly from beta_start to beta_end."""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(num_timesteps, device)

    def _make_betas(self, T):
        return torch.linspace(self.beta_start, self.beta_end, T)


class CosineScheduler(NoiseScheduler):
    """
    Cosine schedule from Nichol & Dhariwal, 'Improved DDPM' (2021).
    Adds noise more gently at the start and end of the process, which tends to
    preserve low-level image detail for longer and is the basis of our
    'noise schedule' ablation.
    """

    def __init__(self, num_timesteps=1000, s=0.008, device="cpu"):
        self.s = s
        super().__init__(num_timesteps, device)

    def _make_betas(self, T):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alpha_bars = torch.cos(((x / T) + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clamp(betas, 0.0001, 0.999)


def get_scheduler(name: str, num_timesteps: int = 1000, device="cpu") -> NoiseScheduler:
    name = name.lower()
    if name == "linear":
        return LinearScheduler(num_timesteps=num_timesteps, device=device)
    elif name == "cosine":
        return CosineScheduler(num_timesteps=num_timesteps, device=device)
    else:
        raise ValueError(f"Unknown schedule '{name}'. Use 'linear' or 'cosine'.")
