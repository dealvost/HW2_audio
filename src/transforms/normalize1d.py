# src/transforms/normalize1d.py
import torch
from torch import nn

class Normalize1D(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
