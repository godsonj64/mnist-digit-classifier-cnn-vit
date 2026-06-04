import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_checkpoint, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["device"])

    _, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    print(f"accuracy={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
