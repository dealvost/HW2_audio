# template_asr_third/src/transforms/wav_augs/time_stretch.py

import torch
from torch import nn
import random

class TimeStretch(nn.Module):
    """
    Заглушка, чтобы убрать ошибку AttributeError,
    если в вашей версии torch_audiomentations нет TimeStretch.
    
    Можно оставить пуское применение (т. е. не менять сигнал).
    Либо использовать другое решение (скорость можно менять через sox_effects).
    """

    def __init__(self, min_rate=0.85, max_rate=1.15, p=0.5, sample_rate=16000):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
        self.sample_rate = sample_rate

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data shape: [1, num_samples].
        Возвращаем data как есть, без изменения.
        """
        # Если хотите, можно добавить случайную задержку / растяжение через torchaudio.sox_effects
        # Но здесь просто делаем заглушку, чтобы не падать с ошибкой.
        return data
