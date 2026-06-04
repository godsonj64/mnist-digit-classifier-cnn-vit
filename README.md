# MNIST Digit Classifier (CNN + ViT)

This project trains an image classifier to recognize handwritten digits (0 through 9)
using the classic MNIST dataset.

It offers two models:

- **Baseline**: a small CNN trained from scratch (fast, good sanity check).
- **Recommended**: an EfficientNet-B0 transfer-learning model. (You may also
  select a hybrid `cnn_vit` model that pairs a small CNN with a Vision Transformer.)

We report **accuracy** and **F1 score** so you know exactly how reliable it is, and
we export the trained model to **ONNX** and **TorchScript** so you can plug it into
your own app.

## Dataset format

The dataset uses the standard `ImageFolder` layout. See [`data/README.md`](data/README.md).

```
data/
  train/
    0/ img1.png ...
    1/ ...
    ...
    9/ ...
  val/
    0/ ...
    ...
```

If no dataset is present, the training script will automatically download MNIST and
lay it out in this format for you.

## Quick start

```bash
pip install -r requirements.txt

# Train (uses configs/default.yaml)
bash scripts/run_train.sh

# Evaluate a trained checkpoint
python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best.pt

# Export to ONNX + TorchScript
python -m src.export --config configs/default.yaml --checkpoint outputs/best.pt
```

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml). Change `model.name`
to one of `baseline_cnn`, `efficientnet_b0`, or `cnn_vit`.

## Training output

The training loop prints one parseable line per epoch:

```
epoch 1/20 loss=0.4213 val_acc=0.9123
```

## Docker

```bash
docker build -t mnist-classifier .
docker run --rm -v $(pwd)/outputs:/app/outputs mnist-classifier
```
