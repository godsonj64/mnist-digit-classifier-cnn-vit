import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_transform(image_size):
    """Resize to a square, convert to 3-channel RGB, and normalize pixel values."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _has_imagefolder_layout(root):
    """Check that data/train and data/val exist with class subfolders."""
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        return False
    return any(os.path.isdir(os.path.join(train_dir, d)) for d in os.listdir(train_dir))


def _download_mnist_as_imagefolder(root):
    """Download MNIST and write PNGs into the ImageFolder layout."""
    tmp = os.path.join(root, "_mnist_raw")
    os.makedirs(tmp, exist_ok=True)
    splits = {
        "train": datasets.MNIST(tmp, train=True, download=True),
        "val": datasets.MNIST(tmp, train=False, download=True),
    }
    for split, ds in splits.items():
        for cls in range(10):
            os.makedirs(os.path.join(root, split, str(cls)), exist_ok=True)
        counters = {c: 0 for c in range(10)}
        for img, label in ds:
            label = int(label)
            out_path = os.path.join(root, split, str(label), f"{counters[label]:05d}.png")
            img.save(out_path)
            counters[label] += 1


def build_dataloaders(config):
    """Create train and validation data loaders, downloading MNIST if needed."""
    data_cfg = config["data"]
    root = data_cfg["root"]

    if not _has_imagefolder_layout(root):
        if data_cfg.get("auto_download", True):
            print(f"No dataset found at '{root}'. Downloading MNIST...")
            _download_mnist_as_imagefolder(root)
        else:
            raise FileNotFoundError(
                f"Expected ImageFolder layout under '{root}' (train/ and val/)."
            )

    transform = _build_transform(data_cfg["image_size"])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
