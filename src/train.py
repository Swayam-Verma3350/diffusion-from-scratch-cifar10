"""
train.py
--------
Fine-tunes the pretrained DDPM (google/ddpm-cifar10-32) on CIFAR-10.

Usage examples:
    python train.py --schedule linear --epochs 5
    python train.py --schedule cosine --epochs 5
    python train.py --classes automobile --epochs 10 --lr 3e-5  # class-specific run

Run on Colab with a T4 GPU. Each run saves:
    - a checkpoint (unet weights)
    - the loss history as a .json for later comparison across ablations
"""

import argparse
import json
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DiffusionModel
from data_utils import get_dataloader


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DiffusionModel(
        schedule=args.schedule,
        num_timesteps=args.num_timesteps,
        pretrained=True,
        device=device,
    )
    model.train()

    classes = [args.classes] if args.classes else None
    loader = get_dataloader(
        root=args.data_root, batch_size=args.batch_size, train=True,
        download=True, classes=classes,
    )

    optimizer = AdamW(model.unet.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine LR decay: smoothly reduces lr to near-zero by the last step.
    # This eliminates the loss oscillation seen with a flat lr and lets the
    # model converge more cleanly in the final epochs.
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)

    loss_history = []
    run_name = f"{args.schedule}" + (f"_{args.classes}" if args.classes else "")
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (images, _) in enumerate(loader):
            images = images.to(device)

            loss = model.training_loss(images)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping: prevents the occasional large gradient spike
            # from destabilizing the pretrained weights (especially important
            # for class-specific fine-tuning with a higher lr).
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # step every batch, not every epoch

            running_loss += loss.item()
            loss_history.append(loss.item())

            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[{run_name}] epoch {epoch} step {step} "
                      f"loss {loss.item():.4f}  lr {current_lr:.2e}")

        avg = running_loss / max(1, len(loader))
        print(f"[{run_name}] epoch {epoch} avg loss {avg:.4f}")

    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min")

    ckpt_path = os.path.join(args.output_dir, f"unet_{run_name}.pt")
    torch.save(model.unet.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    history_path = os.path.join(args.output_dir, f"loss_{run_name}.json")
    with open(history_path, "w") as f:
        json.dump({"run_name": run_name, "loss_history": loss_history,
                   "elapsed_seconds": elapsed}, f)
    print(f"Saved loss history to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--classes", type=str, default=None,
                         help="restrict to a single CIFAR-10 class, e.g. 'automobile'")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    train(args)
