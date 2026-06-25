"""
evaluate.py
-----------
Lightweight, dependency-light analysis tools (full FID requires a 2048-dim
Inception pool, ~50k samples and a reference statistics file -- overkill for
this assignment). Instead we use two simple, honest proxies that are easy to
explain in the report:

1. Diversity: average pairwise pixel-space distance between generated images.
   Low diversity is a fast, simple signal for mode collapse.
2. Quality proxy: feature distance to real CIFAR-10 images in a pretrained
   Inception-v3 embedding space (a simplified, small-sample version of the
   idea behind FID -- we call it "Inception distance" to avoid overclaiming
   it's true FID, which needs much larger sample sizes to be reliable).
3. compare_loss_curves(): overlays loss curves from different ablation runs
   (e.g. linear vs cosine schedule) for the training-stability discussion.
"""

import json
import glob

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


def diversity_score(images: torch.Tensor) -> float:
    """
    images: (N, C, H, W) tensor in [-1, 1].
    Returns the mean pairwise L2 distance between flattened images.
    Higher = more diverse / less mode collapse.
    """
    n = images.shape[0]
    flat = images.view(n, -1)
    dists = torch.cdist(flat, flat, p=2)
    # exclude the diagonal (distance to self = 0)
    mask = ~torch.eye(n, dtype=torch.bool, device=images.device)
    return dists[mask].mean().item()


def _inception_features(images: torch.Tensor, inception_model, device) -> torch.Tensor:
    """images in [-1, 1] at 32x32 -> resize to 299x299 and run through Inception."""
    images = (images + 1) / 2  # back to [0, 1]
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = images.to(device)
    with torch.no_grad():
        feats = inception_model(images)
    return feats


def inception_distance(real_images: torch.Tensor, fake_images: torch.Tensor, device="cpu") -> float:
    """
    A small-sample, simplified stand-in for FID: distance between mean feature
    vectors of real vs. generated images in Inception-v3 pool space.
    NOTE: with small sample sizes (tens to low hundreds of images) this is a
    noisy estimate -- useful for *relative* comparison between our own ablation
    runs, not as an absolute, paper-comparable FID number.
    """
    weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    inception = torchvision.models.inception_v3(weights=weights, aux_logits=True)
    inception.fc = torch.nn.Identity()  # use pre-logit 2048-d features
    inception.eval().to(device)

    real_feats = _inception_features(real_images, inception, device)
    fake_feats = _inception_features(fake_images, inception, device)

    real_mean = real_feats.mean(dim=0)
    fake_mean = fake_feats.mean(dim=0)
    return torch.norm(real_mean - fake_mean, p=2).item()


def compare_loss_curves(history_glob="outputs/loss_*.json", smooth_window=20, save_path="loss_comparison.png"):
    """Overlay loss curves from multiple ablation runs (e.g. linear vs cosine)."""
    plt.figure(figsize=(8, 5))
    for path in sorted(glob.glob(history_glob)):
        with open(path) as f:
            data = json.load(f)
        losses = data["loss_history"]
        # simple moving average for readability
        smoothed = [
            sum(losses[max(0, i - smooth_window):i + 1]) / len(losses[max(0, i - smooth_window):i + 1])
            for i in range(len(losses))
        ]
        plt.plot(smoothed, label=data["run_name"])

    plt.xlabel("Training step")
    plt.ylabel("MSE loss (smoothed)")
    plt.title("Training loss across ablation runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    # quick demo with random noise so the script can be smoke-tested without a GPU run
    dummy_real = torch.rand(8, 3, 32, 32) * 2 - 1
    dummy_fake = torch.rand(8, 3, 32, 32) * 2 - 1
    print("diversity_score (random images):", diversity_score(dummy_fake))
