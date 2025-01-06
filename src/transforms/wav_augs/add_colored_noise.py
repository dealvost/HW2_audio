# src/transforms/wav_augs/add_colored_noise.py
import torch
from torch import nn
import torch_audiomentations

class AddColoredNoiseWrap(nn.Module):
    def __init__(self, p=0.5, sample_rate=16000, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(
            p=p,
            sample_rate=sample_rate,
            **kwargs
        )

    def forward(self, data: torch.Tensor):
        # data: [1, num_samples]
        x = data.unsqueeze(0)  # [1, 1, num_samples]
        x = self._aug(x)
        x = x.squeeze(0) # обратно [1, num_samples]
        return x
