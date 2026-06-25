"""
sample.py
---------
Generates images from a (fine-tuned) checkpoint using either:
    --method ddpm  : full T-step ancestral sampling (slow, reference quality)
    --method ddim  : fast sampling with a configurable number of steps

This script is what produces the "inference speed vs quality" ablation:
run it multiple times with --method ddim --steps 10/50/200 and compare
both wall-clock time and visual quality.

Usage:
    python sample.py --checkpoint outputs/unet_linear.pt --method ddpm --num_images 16
    python sample.py --checkpoint outputs/unet_linear.pt --method ddim --steps 50 --num_images 16
"""

import argparse
import time

import torch
import matplotlib.pyplot as plt

from model import DiffusionModel
from data_utils import tensor_to_image_grid


def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionModel(schedule=args.schedule, num_timesteps=args.num_timesteps,
                            pretrained=True, device=device)
    if args.checkpoint:
        model.unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded fine-tuned weights from {args.checkpoint}")
    model.eval()

    start = time.time()
    if args.method == "ddpm":
        samples = model.sample_ddpm(num_images=args.num_images)
    else:
        samples = model.sample_ddim(num_images=args.num_images,
                                     num_inference_steps=args.steps, eta=args.eta)
    elapsed = time.time() - start
    print(f"Generated {args.num_images} images with method={args.method} "
          f"in {elapsed:.2f}s ({elapsed/args.num_images:.3f}s/image)")

    grid = tensor_to_image_grid(samples, nrow=int(args.num_images ** 0.5))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(f"{args.method.upper()}"
               + (f" ({args.steps} steps)" if args.method == "ddim" else f" ({args.num_timesteps} steps)"))
    out_path = args.output
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved sample grid to {out_path}")

    return elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="path to fine-tuned unet state_dict; omit to use the raw pretrained model")
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--method", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps only")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity, 0=deterministic")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--output", type=str, default="samples.png")
    args = parser.parse_args()

    generate(args)
