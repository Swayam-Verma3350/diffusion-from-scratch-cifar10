"""
data_utils.py
-------------
CIFAR-10 loading + the "Data Exploration" deliverable: class distribution,
basic image statistics, and sample visualization.

Diffusion models work on images scaled to [-1, 1] (matching the tanh-like
output range typically used), not [0, 1], so our transform reflects that.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])


def get_cifar10(root="./data", train=True, download=True):
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, transform=get_transform()
    )


def get_dataloader(root="./data", batch_size=128, train=True, download=True,
                    classes=None, num_workers=2):
    """
    classes: optional list of class names (e.g. ["automobile"]) to restrict
    training to a subset -- used for the 'class-specific training' ablation.
    """
    dataset = get_cifar10(root=root, train=train, download=download)

    if classes is not None:
        target_idxs = [CIFAR10_CLASSES.index(c) for c in classes]
        keep = [i for i, (_, label) in enumerate(dataset) if label in target_idxs]
        dataset = Subset(dataset, keep)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, drop_last=True)


def analyze_class_distribution(root="./data", download=True):
    """Returns and plots a bar chart of how many images belong to each class."""
    dataset = get_cifar10(root=root, train=True, download=download)
    labels = np.array(dataset.targets)
    counts = [int((labels == i).sum()) for i in range(10)]

    plt.figure(figsize=(8, 4))
    plt.bar(CIFAR10_CLASSES, counts)
    plt.xticks(rotation=45)
    plt.ylabel("Number of training images")
    plt.title("CIFAR-10 class distribution")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=120)
    plt.show()

    return dict(zip(CIFAR10_CLASSES, counts))


def show_sample_grid(dataset, n=16, title="Sample images", filename=None):
    """Plot an n-image grid (denormalized back to [0,1] for display)."""
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    fig, axes = plt.subplots(4, n // 4, figsize=(n // 4 * 1.5, 6))
    for ax, idx in zip(axes.flatten(), idxs):
        img, label = dataset[idx]
        img = (img * 0.5 + 0.5).permute(1, 2, 0).numpy()  # back to [0,1], HWC
        ax.imshow(img)
        ax.set_title(CIFAR10_CLASSES[label], fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=120)
    plt.show()


def tensor_to_image_grid(tensor, nrow=4):
    """Utility for turning a batch of model output ([-1,1] range) into a
    numpy image grid for plotting with matplotlib."""
    grid = torchvision.utils.make_grid((tensor + 1) / 2, nrow=nrow)  # back to [0,1]
    return grid.permute(1, 2, 0).cpu().numpy()
