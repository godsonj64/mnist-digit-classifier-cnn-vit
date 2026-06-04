import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Read the YAML settings file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make runs reproducible by fixing all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str):
    """Decide whether to use GPU or CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def ensure_dir(path):
    """Create a folder if it does not exist yet."""
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(model, config, path):
    """Save model weights plus the config used to build it."""
    ensure_dir(os.path.dirname(path) or ".")
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path, map_location="cpu"):
    """Load a saved checkpoint dictionary from disk."""
    return torch.load(path, map_location=map_location)
