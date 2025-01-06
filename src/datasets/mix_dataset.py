import torch
from torch.utils.data import ConcatDataset
from hydra.utils import instantiate


class MixDataset(ConcatDataset):
    """
    Объединяет несколько датасетов (Librispeech, CommonVoice, ...) в один.
    Пример использования в multi.yaml:
    
    train:
      _target_: src.datasets.mix_dataset.MixDataset
      datasets:
        - _target_: src.datasets.LibrispeechDataset
          part: "train-clean-360"
        - _target_: src.datasets.LibrispeechDataset
          part: "train-other-500"
        - _target_: src.datasets.CommonVoiceDataset
          split: "train"
    ...
    
    Таким образом train будет объединением (concatenate) нескольких датасетов.
    """

    def __init__(self, datasets):
        """
        datasets — это список УЖЕ инициализированных датасетов,
        заданных в конфиге.
        """
        super().__init__(datasets)
