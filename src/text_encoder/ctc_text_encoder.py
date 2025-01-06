import re
import torch

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Создание энкодера для CTC-модели.

        :param alphabet: список символов (без blank).
                         Если None — используем a-z + пробел (lower-case).
        """
        if alphabet is None:
            # Переключаемся на нижний регистр и пробел.
            # 26 букв a-z и ' ' (пробел) => итого 27 символов
            import string
            alphabet = list(string.ascii_lowercase + " ")

        self.alphabet = alphabet
        # vocab = [EMPTY_TOK] + alphabet
        self.vocab = [self.EMPTY_TOK] + self.alphabet

        # Пример: ind2char[0] = '', ind2char[1] = 'a', ind2char[2] = 'b', ...
        self.ind2char = dict(enumerate(self.vocab))
        # char2ind['a'] = 1, ...
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        """Возвращает кол-во токенов (включая blank=0)."""
        return len(self.vocab)

    def __getitem__(self, index: int):
        """Позволяет обращаться к энкодеру как к массиву."""
        assert isinstance(index, int), "Index must be int"
        return self.ind2char[index]

    def encode(self, text: str) -> torch.Tensor:
        """
        Преобразуем строку в тензор индексов (учитывая blank=0).
        text -> lower -> filter -> [char2ind].
        """
        text = self.normalize_text(text)
        indices = [self.char2ind[c] for c in text]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Декодирование без учёта CTC постобработки.
        inds: список int
        """
        return "".join(self.ind2char[int(i)] for i in inds)

    def ctc_decode(self, inds) -> str:
        """
        CTC decode: убираем подряд дублирующиеся индексы (кроме 0),
        а также игнорируем blank (0).
        """
        decoded = []
        prev = None
        for i in inds:
            if i != 0 and i != prev:
                decoded.append(i)
            prev = i
        return self.decode(decoded).strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Приводим к нижнему регистру,
        вырезаем всё кроме [a-z и пробела].
        """
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


