import argparse
import os

import torch

from src.model import build_model
from src.utils import ensure_dir, load_checkpoint, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser(description="Export a trained model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["device"])

    model = build_model(config).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image_size = config["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    out_dir = ensure_dir(config["export"]["output_dir"])
    formats = config["export"]["formats"]

    if "onnx" in formats:
        onnx_path = os.path.join(out_dir, "model.onnx")
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
        print(f"Exported ONNX to {onnx_path}")

    if "torchscript" in formats:
        ts_path = os.path.join(out_dir, "model.ts")
        scripted = torch.jit.trace(model, dummy)
        scripted.save(ts_path)
        print(f"Exported TorchScript to {ts_path}")


if __name__ == "__main__":
    main()
