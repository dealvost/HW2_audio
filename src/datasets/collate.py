

import torch

def collate_fn(dataset_items: list[dict]):
    # Найдем макс длину по spectrogram time
    # spectrogram: [B, freq, time]
    # text_encoded: [B, seq_len]
    spectrograms = [x["spectrogram"] for x in dataset_items]

# Убираем измерение канала, если оно есть
    for i in range(len(spectrograms)):
    # Если спектрограмма имеет форму [1, freq, time], то squeeze(0) превратит ее в [freq, time]
        if spectrograms[i].dim() == 3 and spectrograms[i].shape[0] == 1:
            spectrograms[i] = spectrograms[i].squeeze(0)

    texts = [x["text"] for x in dataset_items]
    text_encoded = [x["text_encoded"].squeeze(0) for x in dataset_items]  # each: [seq_len]
    audio_paths = [x["audio_path"] for x in dataset_items]

    # длины по time у спектров
    spec_lengths = torch.tensor([s.shape[-1] for s in spectrograms], dtype=torch.int64)
    max_spec_len = spec_lengths.max().item()

    # длины text_encoded
    text_lengths = torch.tensor([t.shape[0] for t in text_encoded], dtype=torch.int64)
    max_text_len = text_lengths.max().item()

    # паддим спектры до max_spec_len
    # spectrogram shape after stack: [B, freq, time]
    freq = spectrograms[0].shape[0]
    padded_specs = torch.zeros(len(spectrograms), freq, max_spec_len, dtype=spectrograms[0].dtype)
    for i, s in enumerate(spectrograms):
        length = s.shape[-1]
        padded_specs[i, :, :length] = s

    # паддим text_encoded
    padded_texts = torch.zeros(len(text_encoded), max_text_len, dtype=text_encoded[0].dtype, requires_grad=False)
    for i, t in enumerate(text_encoded):
        length = t.shape[0]
        padded_texts[i, :length] = t

    batch = {
        "spectrogram": padded_specs,  # [B, freq, time]
        "spectrogram_length": spec_lengths, 
        "text": texts,
        "text_encoded": padded_texts,  # [B, max_text_len]
        "text_encoded_length": text_lengths,
        "audio_path": audio_paths
    }

    return batch
