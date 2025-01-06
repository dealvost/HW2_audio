# src/model/conformer_model.py
import torch
import torch.nn as nn
from torchaudio.models import Conformer

class ConformerASR(nn.Module):
    def __init__(self, n_feats, n_tokens, encoder_dim=256, num_layers=12, num_attention_heads=4):
        super().__init__()
        # Conformer возвращает тензор размером [B, T, input_dim], где input_dim = n_feats.
        # Здесь n_feats = 128, значит выход будет [B,T,128].
        self.conformer = Conformer(
            input_dim=n_feats,
            num_heads=num_attention_heads,
            ffn_dim=4 * encoder_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        # Задаём линейный слой с in_features=128, соответствуя выходу Conformer
        self.output_linear = nn.Linear(128, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # Исходно spectrogram: [B, F, T]
        # Нужно получить [B, T, F]
        x = spectrogram.permute(0, 2, 1)  # [B,F,T] -> [B,T,F]

        lengths = spectrogram_length  # длины формы [B]
        # Переносим lengths на тот же девайс, что и x
        lengths = lengths.to(x.device)

        # Передаем x (B,T,F) и lengths в conformer
        output, lengths = self.conformer(x, lengths=lengths)  # output: [B,T,128]

        output = self.output_linear(output)  # [B,T,n_tokens]
        log_probs = torch.log_softmax(output, dim=-1)  # [B,T,n_tokens]
        return {"log_probs": log_probs, "log_probs_length": lengths}

    def transform_input_lengths(self, input_lengths):
        return input_lengths

