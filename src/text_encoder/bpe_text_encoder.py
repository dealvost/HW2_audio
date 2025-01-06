import re
import torch
from torch import Tensor
import sentencepiece as spm

class BPETextEncoder:
    """
    BPE-токенизатор для CTC-модели.
    Индекс 0 используем под blank, а реальные BPE-токены идут с 1 до vocab_size-1.
    """

    def __init__(self, model_file="bpe.model"):
        # Загружаем обученную SentencePiece-модель
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file)

        # blank = 0, а все токены из sp идут с 1...
        self.blank_idx = 0
        self.vocab_size = self.sp.GetPieceSize() + 1  # +1 под blank

    def __len__(self):
        return self.vocab_size

    def normalize_text(self, text: str) -> str:
        """
        Минимальная нормализация (lower, trim). 
        Можно усложнить по аналогии с CTCTextEncoder, если нужно.
        """
        text = text.lower().strip()
        return text

    def encode(self, text: str) -> Tensor:
        """
        Превращаем текст в список индексов BPE. 
        Blank=0, а реальные BPE-токены идут c 1.
        """
        text = self.normalize_text(text)
        pieces = self.sp.EncodeAsIds(text)  # list of int
        # Сдвигаем все на +1
        shifted = [p + 1 for p in pieces]
        return torch.tensor(shifted, dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Декодирование без пост-обработки CTC. 
        Убираем blank=0, затем смещаем на -1, чтобы получить valid BPE ids.
        Приводим к int, чтобы DecodeIds понимал.
        """
        # отфильтруем blank=0
        filtered = []
        for i in inds:
            if i != 0:
                bpe_id = i - 1  # сместить назад
                # убедимся, что bpe_id >= 0
                if bpe_id >= 0:
                    filtered.append(int(bpe_id))

        if len(filtered) == 0:
            return ""
        return self.sp.DecodeIds(filtered)

    def ctc_decode(self, inds) -> str:
        """
        CTC decode: убираем подряд идущие дубликаты (кроме blank=0) и игнорируем blank=0.
        """
        decoded = []
        prev = None
        for i in inds:
            if i != 0 and i != prev:
                decoded.append(i)
            prev = i
        return self.decode(decoded).strip()

