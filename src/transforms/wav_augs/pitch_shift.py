# template_asr_third/src/transforms/wav_augs/pitch_shift.py

import torch
from torch import nn
import torch_audiomentations as taa

class PitchShift(nn.Module):
    """
    Случайный pitch-shift с помощью torch_audiomentations.
    Двигает pitch в диапазоне +/- n полутонов.
    """

    def __init__(self, min_transpose_semitones=-4.0, max_transpose_semitones=4.0, p=0.5, sample_rate=16000):
        super().__init__()
        self._aug = taa.PitchShift(
            sample_rate=sample_rate,
            p=p,
            min_transpose_semitones=min_transpose_semitones,
            max_transpose_semitones=max_transpose_semitones,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [1, num_samples]
        Возвращаем тот же shape: [1, num_samples].
        """
        x = data.unsqueeze(0)   # -> [B=1, 1, num_samples]
        x = self._aug(x)
        x = x.squeeze(0)        # -> [1, num_samples]
        return x
