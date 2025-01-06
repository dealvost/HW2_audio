import os
import torch
import kenlm   # pip install https://github.com/kpu/kenlm/archive/master.zip
import pyctcdecode  # pip install pyctcdecode
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from hydra.utils import to_absolute_path

from src.utils.init_utils import set_random_seed
from src.datasets.data_utils import get_dataloaders
from src.metrics.utils import calc_wer

@hydra.main(version_base=None, config_path="src/configs", config_name="inference_BPE")
def main(config):


    # 1) Seed + device
    set_random_seed(config.inferencer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 2) Инициализируем BPETextEncoder и PyTorch-модель
    text_encoder = instantiate(config.text_encoder)  # BPETextEncoder
    n_tokens = len(text_encoder)  # blank + subword tokens

    model = instantiate(config.model, n_tokens=n_tokens).to(device)
    ckpt_path = config.inferencer.get("from_pretrained", None)
    if not ckpt_path:
        raise ValueError("No from_pretrained path in config!")
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 3) Готовим labels для pyctcdecode.
    # blank_id = 0, а все реальные BPE токены идут с 1...n-1
    # pyctcdecode требует list[str] вида:
    #   labels[0] = '', labels[1] = '▁the', labels[2]='▁he', ...
    #   (Ещё нужен символ для ' '?), но мы "попробуем" работать как есть.
    bpe_labels = []
    for idx in range(n_tokens):
        if idx == 0:
            bpe_labels.append("")  # blank
        else:
            # decодируем "виртуальный" id: idx -> bpe id (idx-1)
            # внутри text_encoder.sp.IdToPiece(...)
            piece_str = text_encoder.sp.IdToPiece(idx - 1)
            bpe_labels.append(piece_str)

    print("Example of BPE labels:", bpe_labels[:30])

    # 3б) Грузим ARPA LM. 
    # pyctcdecode.build_ctcdecoder позволяет автоматически создать decoder
    #   c использованием KenLM.
    # unigrams: можете загрузить словарь (lower-case):
    vocab_path = to_absolute_path("librispeech-vocab_lower.txt")  # если он есть
    if not os.path.exists(vocab_path):
        print(f"Warning: no {vocab_path}. We'll skip 'unigrams'.")
        unigrams = None
    else:
        with open(vocab_path, "r", encoding="utf-8") as f:
            unigrams = [w.strip() for w in f if w.strip()]

    lm_path = to_absolute_path("4-gram-lower.arpa")
    if not os.path.exists(lm_path):
        raise FileNotFoundError("No 4-gram.arpa found at /content/4-gram.arpa")

    decoder = pyctcdecode.build_ctcdecoder(
        labels=bpe_labels,
        kenlm_model_path=lm_path,
        unigrams=unigrams,  # может быть None
        alpha=0.5,   # коэффициент LM
        beta=1.5,    # словоразделитель
        # если хотите, можно настроить другие параметры
    )

    # 4) Готовим dataloader
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)
    test_loader = dataloaders.get("test") or dataloaders.get("val")
    if not test_loader:
        raise ValueError("No test/val dataloader found. Check config.datasets")

    print("Start Beam Search with pyctcdecode + KenLM...")

    total_wer = 0.0
    total_count = 0

    # 5) Инференс
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="InferenceLMBeam"):
            spectrogram = batch["spectrogram"].to(device)
            spectrogram_length = batch["spectrogram_length"].to(device)

            outputs = model(spectrogram, spectrogram_length)
            log_probs = outputs["log_probs"]  # [B,T,n_tokens]
            log_probs_length = outputs["log_probs_length"]  # [B]

            # pyctcdecode хочет cpu float numpy shape [time, vocab_size]
            # А у нас [B, time, vocab_size]. Пройдёмся по батчу:
            log_probs_np = log_probs.cpu().float().numpy()  # [B, T, V]
            lengths_np = log_probs_length.cpu().numpy()

            for i in range(len(log_probs_np)):
                T_i = int(lengths_np[i])
                # Вырежем нужную часть
                logits_i = log_probs_np[i, :T_i, :]  # shape (Ti, n_tokens)
                # beam search decode
                beam_str = decoder.decode(logits_i, beam_width=100)  # можно beam_width=50 etc.

                # Сравниваем с GT
                target_str = text_encoder.normalize_text(batch["text"][i])
                wer = calc_wer(target_str, beam_str)
                total_wer += wer
                total_count += 1

                if total_count <= 5:
                    print(f"GT: {target_str}")
                    print(f"Decoded with LM beam: {beam_str}")
                    print(f"WER={wer}\n---")

    avg_wer = total_wer / total_count if total_count > 0 else 999
    print(f"Finished. Average WER beam+LM = {avg_wer*100:.2f}%")

if __name__ == "__main__":
    main()

