# src/trainer/base_trainer.py

from abc import abstractmethod
import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        text_encoder,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            text_encoder (CTCTextEncoder): text encoder.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # old monitoring logic
        self.monitor = self.cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            # self.mnt_best = 0 # not used now, we have separate metrics
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()

        # New logic: track best WER and CER separately
        self.mnt_best_wer = inf
        self.mnt_best_cer = inf

        self.save_period = self.cfg_trainer.save_period
        self.early_stop = self.cfg_trainer.get("early_stop", inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, filename="checkpoint-last.pth", save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # Check for improvements in WER and CER (val_WER_(Argmax), val_CER_(Argmax))
            current_val_wer = logs.get("val_WER_(Argmax)", None)
            current_val_cer = logs.get("val_CER_(Argmax)", None)

            wer_improved = False
            cer_improved = False

            if current_val_wer is not None and current_val_wer < self.mnt_best_wer:
                self.mnt_best_wer = current_val_wer
                wer_improved = True

            if current_val_cer is not None and current_val_cer < self.mnt_best_cer:
                self.mnt_best_cer = current_val_cer
                cer_improved = True

            # Если улучшился WER -> сохраняем model_best_wer.pth
            if wer_improved:
                self._save_checkpoint(epoch, filename="model_best_wer.pth", save_best=False)
                self.logger.info("Saving current best by WER: model_best_wer.pth ...")

            # Если улучшился CER -> сохраняем model_best_cer.pth
            if cer_improved:
                self._save_checkpoint(epoch, filename="model_best_cer.pth", save_best=False)
                self.logger.info("Saving current best by CER: model_best_cer.pth ...")

            # Если необходимо - логика early_stop по основному monitor
            if self.monitor != "off":
                # Можно оставить или доработать при необходимости.
                metric_value = logs.get(self.mnt_metric, None)
                improved = False
                if metric_value is not None:
                    if self.mnt_mode == "min" and metric_value < self.mnt_best_wer: # пример
                        improved = True
                    # при желании можно усложнить логику

                if improved:
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count >= self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            # сохранить чекпоинт каждые save_period эпох
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, filename=f"checkpoint-epoch{epoch}.pth", save_best=False)
                self.logger.info(f"Saving checkpoint: checkpoint-epoch{epoch}.pth ...")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        last_train_metrics = {}
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _save_checkpoint(self, epoch, filename="model_best.pth", save_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            filename (str): the name of the checkpoint file
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best_wer": self.mnt_best_wer,
            "monitor_best_cer": self.mnt_best_cer,
            "config": self.config,
        }
        save_path = self.checkpoint_dir / filename
        torch.save(state, save_path)
        if self.config.writer.log_checkpoints:
            self.writer.add_checkpoint(str(save_path), str(self.checkpoint_dir.parent))

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch.
        """
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](batch[transform_name])
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch.
        """
        raise NotImplementedError()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best_wer = checkpoint.get("monitor_best_wer", inf)
        self.mnt_best_cer = checkpoint.get("monitor_best_cer", inf)

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.
        """
        pretrained_path = str(pretrained_path)
        self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
