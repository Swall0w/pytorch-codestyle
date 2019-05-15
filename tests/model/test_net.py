from mnist.model import Net
import torch
import pytest


def test_Net():
    model = Net()
    inputs = torch.randn(1, 1, 28, 28)
    out = model(inputs) # torch.Size([1, 10])

    assert out.shape == (1, 10)
