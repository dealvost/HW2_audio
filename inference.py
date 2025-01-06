# inference.py
import warnings
import hydra
import torch
import kenlm
import pyctcdecode
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path

import os
import logging
import numpy as np

from src.datasets.data_utils import get_dataloaders
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


class SameAsValInferencer(BaseTrainer):
  

    def __init__(
        self,
        model,
        criterion,
        metrics,
        text_encoder,
        config,
        device,
        dataloaders,
        logger,
        writer=None,
        batch_transforms=None,
        skip_oom=True,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=None,  # нет оптимизатора
            lr_scheduler=None,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=None,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )
        self.is_train = False

        
        if self.metrics is not None and len(self.metrics["inference"]) > 0:
            metric_names = [m.name for m in self.metrics["inference"]]
            
            metric_names += ["CER_(LM)", "WER_(LM)"]
            self.evaluation_metrics = MetricTracker(*metric_names, writer=writer)
        else:
            
            self.evaluation_metrics = MetricTracker("CER_(LM)", "WER_(LM)", writer=writer)

        # Создаём decoder c LM (теперь с возможностью использовать словарь)
        self.decoder = self._build_decoder_with_lm()

    def _build_decoder_with_lm(self):
      
        # Собираем labels: blank=0 => ""
        ctc_labels = []
        for i in range(len(self.text_encoder)):
            if i == 0:
                ctc_labels.append("")  # blank
            else:
                ctc_labels.append(self.text_encoder.ind2char[i])

        self.logger.info("Building pyctcdecode decoder...")

        # Путь к LM
        lm_path = to_absolute_path("4-gram-lower.arpa")  
        self.logger.info(f"Loading KenLM from: {lm_path}")

        # Путь к словарю
        dict_path = to_absolute_path("librispeech-vocab_lower.txt")  # поменяйте при необходимости
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                unigrams = [w.strip() for w in f if w.strip()]
            self.logger.info(f"Loaded dictionary from {dict_path}, total unigrams: {len(unigrams)}")
        else:
            unigrams = None
            self.logger.warning(f"No dictionary found at {dict_path}. Decoder will work without unigrams.")

        decoder = pyctcdecode.build_ctcdecoder(
            labels=ctc_labels,
            kenlm_model_path=lm_path,
            alpha=0.5,   # вес LM
            beta=1.0,    # штраф за слово
            unigrams=unigrams,  # передаём словарь (или None, если не найден)
        )
        return decoder

    def run_inference(self):
     
        for part_name, dataloader in self.evaluation_dataloaders.items():
            self.logger.info(f"=== Inference on partition: {part_name} ===")
            self.is_train = False
            self.model.eval()

            self.evaluation_metrics.reset()

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        self.process_batch(batch, metrics=self.evaluation_metrics)
                    except torch.cuda.OutOfMemoryError:
                        if self.skip_oom:
                            self.logger.warning("OOM during inference. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

            result = self.evaluation_metrics.result()
            for k, v in result.items():
                self.logger.info(f"{part_name}_{k} = {v:.4f}")

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Реализуем beam search + LM, затем считаем CER_(LM), WER_(LM).
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        log_probs = batch["log_probs"]      # [B, T, vocab_size]
        log_probs_length = batch["log_probs_length"]

        log_probs_np = log_probs.detach().cpu().float().numpy()
        lengths_np = log_probs_length.detach().cpu().numpy()

        # beam-search
        predicted_texts = []
        for i in range(len(log_probs_np)):
            t_i = int(lengths_np[i])
            logits_i = log_probs_np[i, :t_i, :]  # shape [time, vocab_size]
            pred_str = self.decoder.decode(logits_i, beam_width=100)
            predicted_texts.append(pred_str)

        # Считаем CER_(LM), WER_(LM) вручную
        from src.metrics.utils import calc_cer, calc_wer

        target_texts = batch["text"]  # GT
        total_cer = 0.0
        total_wer = 0.0
        N = len(predicted_texts)
        for i in range(N):
            ref_str = self.text_encoder.normalize_text(target_texts[i])
            hyp_str = predicted_texts[i]
            c = calc_cer(ref_str, hyp_str)
            w = calc_wer(ref_str, hyp_str)
            total_cer += c
            total_wer += w

        avg_cer = total_cer / N
        avg_wer = total_wer / N

        # Запишем в MetricTracker
        metrics.update("CER_(LM)", avg_cer)
        metrics.update("WER_(LM)", avg_wer)

       
        if self.criterion is not None:
            all_losses = self.criterion(**batch)
            # обновим metrics
            for loss_name, val in all_losses.items():
                metrics.update(loss_name, val.item())

        return batch


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    
    set_random_seed(config.inferencer.seed)

    # device
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    logger = logging.getLogger("inferencer")
    logger.setLevel(logging.INFO)

    # Инициализируем text_encoder
    text_encoder = instantiate(config.text_encoder)

    # dataloader
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # модель
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    logger.info(model)

    # подгрузим checkpoint
    ckpt_path = config.inferencer.get("from_pretrained", None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Loaded checkpoint from: {ckpt_path}")

    # loss
    criterion = None
    if hasattr(config, "loss_function"):
        criterion = instantiate(config.loss_function).to(device)

    # метрики
    metrics = {"train": [], "inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    # инициализируем инференсер
    inferencer = SameAsValInferencer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=None,
        batch_transforms=batch_transforms,
        skip_oom=True,
    )

    inferencer.run_inference()
    logger.info("=== Inference complete! ===")


if __name__ == "__main__":
    main()


