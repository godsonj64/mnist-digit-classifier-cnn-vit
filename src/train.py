import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import (
    ensure_dir,
    load_config,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def evaluate_loader(model, loader, device):
    """Run the model over a loader and return accuracy and macro F1."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    epochs = config["train"]["epochs"]
    output_dir = ensure_dir(config["train"]["output_dir"])
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_acc, val_f1 = evaluate_loader(model, val_loader, device)

        print(f"epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint(model, config, os.path.join(output_dir, "best.pt"))

    save_checkpoint(model, config, os.path.join(output_dir, "last.pt"))
    print(f"Training complete. Best val_acc={best_acc:.4f} (macro F1 last={val_f1:.4f})")


if __name__ == "__main__":
    main()
