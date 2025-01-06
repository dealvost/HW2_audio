import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        logger=None,
        writer=None,
        epoch_len=None,
        skip_oom=True,
    ):
        # Проверка from_pretrained
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.model = model
        self.text_encoder = text_encoder
        self.save_path = save_path
        self.batch_transforms = batch_transforms

        # Проверяем наличие metrics и ключей "train" и "inference"
        if metrics is None:
            metrics = {"train": [], "inference": []}
        else:
            if "train" not in metrics:
                metrics["train"] = []
            if "inference" not in metrics:
                metrics["inference"] = []

        # Проверяем наличие train в dataloaders
        # Если его нет, создаем фиктивный train DataLoader
        if "train" not in dataloaders:
            class EmptyDataset(Dataset):
                def __len__(self):
                    return 1
                def __getitem__(self, idx):
                    return {
                        "audio": torch.randn(1,16000),
                        "spectrogram": torch.randn(128,100),
                        "text":"",
                        "text_encoded": torch.zeros((1,),dtype=torch.long),
                        "audio_path":"none.wav"
                    }
            train_loader = DataLoader(EmptyDataset(), batch_size=1)
            dataloaders["train"] = train_loader

        # evaluation_dataloaders - все кроме train
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

        self.is_train = False

        # инициализируем evaluation_metrics, если есть metrics
        if self.metrics is not None and len(self.metrics["inference"]) > 0:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        batch_size = batch["log_probs"].shape[0]
        current_id = batch_idx * batch_size

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for i in range(batch_size):
            logits = batch["log_probs"][i].clone()
            label = batch["text_encoded"][i].clone()
            pred_label = logits.argmax(dim=-1)
            output_id = current_id + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }
            torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        else:
            return {}

