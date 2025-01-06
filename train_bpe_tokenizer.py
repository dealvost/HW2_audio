import json
import os
from pathlib import Path
import sentencepiece as spm

def gather_texts_for_bpe(data_dir, output_txt="all_texts_for_bpe.txt"):
    data_dir.mkdir(parents=True, exist_ok=True)  # создание директории, если её нет
    
    texts = []
    for json_file in data_dir.glob("*_index.json"):
        with json_file.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        for item in index_data:
            texts.append(item["text"].strip())

    out_path = data_dir / output_txt
    with out_path.open("w", encoding="utf-8") as f:
        for line in texts:
            if line.strip():
                f.write(line + "\n")
    return out_path

def train_bpe_model(input_txt, vocab_size=500, model_prefix="bpe"):
    model_type = "bpe"
    spm.SentencePieceTrainer.Train(
        f"--input={input_txt} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type={model_type} "
        "--unk_id=0 --pad_id=-1 --bos_id=-1 --eos_id=-1 --user_defined_symbols=[SPACE]"
    )

if __name__ == "__main__":
    # ВАЖНО: пропишите корректный путь к librispeech
    # Если запускаете скрипт в /content/template_asr/, то абсолютный путь может быть таким:
    data_dir = Path("/content/template_asr/data/datasets/librispeech")

    output_txt = gather_texts_for_bpe(data_dir, "all_texts_for_bpe.txt")
    print("Собрали тексты в:", output_txt)

    train_bpe_model(output_txt, vocab_size=500, model_prefix="bpe")
    print("BPE модель обучена: bpe.model, bpe.vocab")
