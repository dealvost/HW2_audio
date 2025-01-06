import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class MaskConv2d(nn.Module):
    """
    Последовательная сверточная часть, которая маскирует выход за пределами
    реальной длины (seq_len), чтобы батчи с разной длиной не "заражали" друг друга.
    """

    def __init__(self, conv_layers: nn.Sequential):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x shape: [B, C, F, T] (batch, channel, freq, time)
        seq_lengths: [B]
        """
        out = x
        for layer in self.conv_layers:
            out = layer(out)  # [B, C, F, T'] после слоя
            # Создадим маску (bool) под нужные seq_len (по оси time)
            seq_lengths = self.update_seq_len(layer, seq_lengths)

            mask = torch.BoolTensor(out.shape).fill_(False)
            if out.is_cuda:
                mask = mask.to(out.device)

            for i, length in enumerate(seq_lengths):
                length = int(length.item())
                if out.size(-1) > length:  # маскируем всё, что за пределами time
                    mask[i, :, :, length:] = True

            out = out.masked_fill(mask, 0.0)

        return out, seq_lengths

    def update_seq_len(self, layer: nn.Module, seq_lengths: Tensor) -> Tensor:
        """
        Пересчитываем seq_lengths (по оси time) после Conv2d или MaxPool2d.
        """
        if isinstance(layer, nn.Conv2d):
            # new_length = floor((old_length + 2*pad - (ker-1)*dilation - 1)/stride) + 1
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            # распакуем
            _kh, kw = kernel_size
            _sh, sw = stride
            _ph, pw = padding
            _dh, dw = dilation

            old_len = seq_lengths
            numerator = old_len + 2 * pw - dw * (kw - 1) - 1
            new_len = numerator.float() / float(sw)
            new_len = new_len.floor().int() + 1
            new_len = torch.clamp(new_len, min=0)
            return new_len

        elif isinstance(layer, nn.MaxPool2d):
            return (seq_lengths >> 1).int()

        return seq_lengths


class DS2ConvExtractor(nn.Module):
    """
    Сверточный экстрактор для DeepSpeech2:
    2 сверточных слоя, каждый с batch-norm и активацией (ReLU или Hardtanh).
    out_channels=64 даёт больше параметров, чем 32.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,  # увеличено для ~100M параметров
        activation: str = 'hardtanh',
    ):
        super().__init__()
        if activation == 'hardtanh':
            act = nn.Hardtanh(0, 20, inplace=True)
        elif activation == 'relu':
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(41, 11),
                    stride=(2, 2),
                    padding=(20, 5),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                act,
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                act,
            )
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x shape: [B, T, F]
        seq_lengths: [B]
        """
        # Переставим в conv-friendly формат [B, C=1, F, T]
        x = x.unsqueeze(1).transpose(2, 3)  # (B,1,F,T)
        out, out_seq_len = self.conv(x, seq_lengths)

        # out shape: [B, C, F2, T2]
        bsz, ch, f2, t2 = out.size()
        # Преобразуем в [B, T2, ch*f2]
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, T2, C, F2)
        out = out.view(bsz, t2, ch*f2)  # (B, T2, C*F2)
        return out, out_seq_len

    def output_dim(self, input_freq_dim: int = 128) -> int:
        """
        Предполагаем, что после двух conv'ов остаётся freq//4.
        С out_channels=64 получается 64*(freq//4).
        """
        return self.out_channels * (input_freq_dim // 4)


class BatchRNN(nn.Module):
    """
    RNN слой с batch-norm по входам + ReLU.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = 'gru',
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.batch_norm = nn.BatchNorm1d(input_size)

        if rnn_type.lower() == 'gru':
            rnn_class = nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = nn.LSTM
        else:
            rnn_class = nn.RNN

        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, x: Tensor, seq_lengths: Tensor) -> Tensor:
        """
        x shape: [B, T, F]
        seq_lengths: [B]
        """
        # batch-norm по оси F
        x = x.transpose(1, 2)  # (B,F,T)
        x = F.relu(self.batch_norm(x))
        x = x.transpose(1, 2)  # (B,T,F)

        # Пакуем
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class DeepSpeech2ASR(nn.Module):
    """
    DeepSpeech2 (с ~100 млн параметров):
    - out_channels=64 в свёрточном экстракторе
    - hidden_size=1536, num_rnn_layers=8, bidirectional=True
    """

    def __init__(
        self,
        n_feats: int = 128,       # кол-во мел-бин/features
        n_tokens: int = 28,       # кол-во выходных токенов
        rnn_type: str = 'gru',
        hidden_size: int = 1536,  # увеличиваем до 1536
        num_rnn_layers: int = 8,
        bidirectional: bool = True,
        dropout: float = 0.1,
        activation: str = 'hardtanh',
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size

        # Сверточный экстрактор (увеличен out_channels=64)
        self.conv_extractor = DS2ConvExtractor(
            in_channels=1,
            out_channels=64,
            activation=activation,
        )

        # RNN слои
        self.rnn_layers = nn.ModuleList()
        rnn_in = self.conv_extractor.output_dim(n_feats)  # обычно 64*(128//4)=64*32=2048
        for i in range(num_rnn_layers):
            # dropout включаем начиная со второго слоя
            rnn_layer = BatchRNN(
                input_size=(rnn_in if i == 0 else hidden_size * (2 if bidirectional else 1)),
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout=(dropout if i != 0 else 0.0),
            )
            self.rnn_layers.append(rnn_layer)

        rnn_out = hidden_size * (2 if bidirectional else 1)

        # Финальный линейный слой + LayerNorm
        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_out),
            nn.Linear(rnn_out, n_tokens, bias=True),
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **kwargs):
        """
        Обычно, если вход: [B, freq, time], то x = spectrogram.transpose(1, 2).
        """
        x = spectrogram.transpose(1, 2)  # теперь [B, T, F]

        conv_out, out_lengths = self.conv_extractor(x, spectrogram_length)

        # Прогоняем через RNN слои
        rnn_out = conv_out
        for layer in self.rnn_layers:
            rnn_out = layer(rnn_out, out_lengths)

        # Линейный слой
        logits = self.fc(rnn_out)  # [B,T,n_tokens]
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return {
            "log_probs": log_probs,           # [B, T, n_tokens]
            "log_probs_length": out_lengths,  # обновлённые длины
        }

    def transform_input_lengths(self, input_lengths: Tensor) -> Tensor:
        # По факту всё пересчитывается внутри conv_extractor,
        # поэтому можно просто вернуть input_lengths.
        return input_lengths

