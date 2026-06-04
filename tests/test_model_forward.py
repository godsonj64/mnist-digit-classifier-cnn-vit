import torch

from src.model import build_model


def _base_config(name):
    return {
        "data": {"num_classes": 10, "image_size": 32},
        "model": {"name": name, "pretrained": False},
    }


def test_baseline_cnn_forward():
    model = build_model(_base_config("baseline_cnn")).eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_cnn_vit_forward():
    model = build_model(_base_config("cnn_vit")).eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_efficientnet_forward():
    model = build_model(_base_config("efficientnet_b0")).eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
