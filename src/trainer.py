from dataclasses import dataclass, field
from statistics import fmean

import torch
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm

from .utils import argmin


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "none"  # none | cosine | warmup_cosine
    warmup_epochs: int = 0
    eta_min: float = 0.0  # ratio min_lr / peak_lr


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    grad_accum: int = 1
    max_grad_norm: float = 0.0
    early_stop_patience: int = 0
    early_stop_metric: str = "loss_val"
    early_stop_min_delta: float = 0.0


class Trainer:
    """Single training loop driven entirely by `OptimConfig` + `TrainConfig`.

    Supports classification (CE + accuracy) and regression (MSE + RMSE/MAE/R^2),
    AdamW with optional warmup+cosine schedule, gradient accumulation, gradient
    clipping, and val-loss based early stopping.
    """

    def __init__(self, model, device, logger, task="classification", forward_takes_mask=False):
        self.model = model
        self.device = device
        self.logger = logger
        self.task = task
        self.forward_takes_mask = forward_takes_mask
        self.history = {
            "train": [],
            "val": [],
            "test": [],
            "acc": [],
            "rmse": [],
            "mae": [],
            "r2": [],
            "lr": [],
        }
        self.loss_fn = self._build_loss_fn(task)
        self.optimizer = None

    @staticmethod
    def _build_loss_fn(task):
        if task == "regression":
            return torch.nn.MSELoss()
        return torch.nn.CrossEntropyLoss()

    def _unpack_batch(self, batch):
        if self.forward_takes_mask:
            x, mask, y = batch
            return x.to(self.device), mask.to(self.device), y.to(self.device)
        x, y = batch
        return x.to(self.device), None, y.to(self.device)

    def _forward(self, x, mask):
        if self.forward_takes_mask:
            return self.model(x, mask)
        return self.model(x)

    @staticmethod
    def _prepare_regression_targets(y_hat, y):
        y = y.float().view(y_hat.shape[0], -1)
        if y_hat.ndim == 1:
            y_hat = y_hat.unsqueeze(-1)
        return y_hat, y

    @staticmethod
    def _classification_metrics(y_hat, y):
        return {"acc": float(multiclass_accuracy(y_hat, y).detach().cpu().item())}

    def _regression_metrics(self, y_hat, y):
        y_hat, y = self._prepare_regression_targets(y_hat, y)
        diff = y_hat - y
        mse = torch.mean(diff * diff)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))
        ss_res = torch.sum(diff * diff)
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) * (y - y_mean))
        r2 = 1.0 - (ss_res / ss_tot) if float(ss_tot) > 0.0 else torch.tensor(0.0, device=y.device)
        return {
            "rmse": float(rmse.detach().cpu().item()),
            "mae": float(mae.detach().cpu().item()),
            "r2": float(r2.detach().cpu().item()),
        }

    def _train_step(self, dataloader, train_cfg: TrainConfig, stage):
        loss_batches = []
        accum = max(1, int(train_cfg.grad_accum))
        max_norm = float(train_cfg.max_grad_norm)
        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(dataloader):
            x, mask, y = self._unpack_batch(batch)
            y_hat = self._forward(x, mask)
            if self.task == "regression":
                y_hat, y = self._prepare_regression_targets(y_hat, y)
            loss = self.loss_fn(y_hat, y)
            (loss / accum).backward()
            if (step_idx + 1) % accum == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            loss_batches.append(float(loss.detach().cpu().item()))
        # flush leftover grads if dataset doesn't divide evenly
        if (len(loss_batches) % accum) != 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self.history[stage].append(fmean(loss_batches) if loss_batches else float("nan"))

    def _eval_step(self, dataloader, stage):
        loss_batches = []
        metrics_batches = []
        with torch.no_grad():
            for batch in dataloader:
                x, mask, y = self._unpack_batch(batch)
                y_hat = self._forward(x, mask)
                if self.task == "regression":
                    y_hat, y = self._prepare_regression_targets(y_hat, y)
                loss = self.loss_fn(y_hat, y)
                loss_batches.append(float(loss.detach().cpu().item()))

                if stage == "test":
                    if self.task == "regression":
                        metrics_batches.append(self._regression_metrics(y_hat, y))
                    else:
                        metrics_batches.append(self._classification_metrics(y_hat, y))

        self.history[stage].append(fmean(loss_batches) if loss_batches else float("nan"))
        if stage == "test":
            if self.task == "regression":
                for key in ("rmse", "mae", "r2"):
                    self.history[key].append(fmean([m[key] for m in metrics_batches]))
            else:
                self.history["acc"].append(fmean([m["acc"] for m in metrics_batches]))

    def _build_optimizer(self, cfg: OptimConfig):
        if cfg.optimizer.lower() != "adamw":
            raise ValueError("Only AdamW optimizer is supported (got '{}').".format(cfg.optimizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _build_scheduler(self, cfg: OptimConfig, n_epochs: int):
        scheduler_name = cfg.scheduler.lower()
        if scheduler_name == "none":
            return None
        eta_min_abs = cfg.lr * float(cfg.eta_min)
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, n_epochs),
                eta_min=eta_min_abs,
            )
        if scheduler_name == "warmup_cosine":
            warmup_epochs = int(max(0, cfg.warmup_epochs))
            if warmup_epochs <= 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, n_epochs),
                    eta_min=eta_min_abs,
                )
            if warmup_epochs >= n_epochs:
                return torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0 / warmup_epochs,
                    end_factor=1.0,
                    total_iters=max(1, n_epochs),
                )
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0 / warmup_epochs,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, n_epochs - warmup_epochs),
                eta_min=eta_min_abs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        raise ValueError("Unknown scheduler '{}'.".format(cfg.scheduler))

    def _postfix(self, epoch_idx, current_lr=None):
        base = (
            "t={:.4f}, t*={:.4f}, v={:.4f}, v*={:.4f}, ts={:.4f}, ts*={:.4f}, "
            "ts@t*={:.4f}, ts@v*={:.4f}"
        ).format(
            self.history["train"][epoch_idx],
            min(self.history["train"][: epoch_idx + 1]),
            self.history["val"][epoch_idx],
            min(self.history["val"][: epoch_idx + 1]),
            self.history["test"][epoch_idx],
            min(self.history["test"][: epoch_idx + 1]),
            self.history["test"][argmin(self.history["train"][: epoch_idx + 1])],
            self.history["test"][argmin(self.history["val"][: epoch_idx + 1])],
        )
        if self.task == "regression":
            metric = (
                ", rmse={:.4f}, rmse*={:.4f}, rmse@v*={:.4f}, mae={:.4f}, r2={:.4f}"
            ).format(
                self.history["rmse"][epoch_idx],
                min(self.history["rmse"][: epoch_idx + 1]),
                self.history["rmse"][argmin(self.history["val"][: epoch_idx + 1])],
                self.history["mae"][epoch_idx],
                self.history["r2"][epoch_idx],
            )
        else:
            metric = ", acc={:.4f}, acc*={:.4f}, acc@t*={:.4f}, acc@v*={:.4f}".format(
                self.history["acc"][epoch_idx],
                max(self.history["acc"][: epoch_idx + 1]),
                self.history["acc"][argmin(self.history["train"][: epoch_idx + 1])],
                self.history["acc"][argmin(self.history["val"][: epoch_idx + 1])],
            )
        if current_lr is not None:
            metric += ", lr={:.2e}".format(current_lr)
        return base + metric

    def _build_log_metrics(self, epoch_idx, current_lr=None):
        payload = {
            "loss_train": self.history["train"][epoch_idx],
            "loss_train_best": min(self.history["train"][: epoch_idx + 1]),
            "loss_val": self.history["val"][epoch_idx],
            "loss_val_best": min(self.history["val"][: epoch_idx + 1]),
            "loss_test": self.history["test"][epoch_idx],
            "loss_test_best": min(self.history["test"][: epoch_idx + 1]),
            "loss_test_at_train_best": self.history["test"][argmin(self.history["train"][: epoch_idx + 1])],
            "loss_test_at_val_best": self.history["test"][argmin(self.history["val"][: epoch_idx + 1])],
        }
        if self.task == "regression":
            payload.update(
                {
                    "rmse_test": self.history["rmse"][epoch_idx],
                    "rmse_test_best": min(self.history["rmse"][: epoch_idx + 1]),
                    "rmse_test_at_val_best": self.history["rmse"][argmin(self.history["val"][: epoch_idx + 1])],
                    "mae_test": self.history["mae"][epoch_idx],
                    "r2_test": self.history["r2"][epoch_idx],
                }
            )
        else:
            payload.update(
                {
                    "acc_test": self.history["acc"][epoch_idx],
                    "acc_test_best": max(self.history["acc"][: epoch_idx + 1]),
                    "acc_test_at_train_best": self.history["acc"][argmin(self.history["train"][: epoch_idx + 1])],
                    "acc_test_at_val_best": self.history["acc"][argmin(self.history["val"][: epoch_idx + 1])],
                }
            )
        if current_lr is not None:
            payload["lr"] = current_lr
        return payload

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        *,
        optim_config: OptimConfig,
        train_config: TrainConfig,
        desc: str | None = None,
    ):
        self._build_optimizer(optim_config)
        self.model.to(self.device)
        scheduler_obj = self._build_scheduler(optim_config, train_config.epochs)

        if optim_config.scheduler.lower() == "warmup_cosine" and optim_config.warmup_epochs > 0:
            initial_lr = optim_config.lr * (1.0 / optim_config.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = initial_lr

        early_stop_patience = int(max(0, train_config.early_stop_patience))
        early_stop_min_delta = float(max(0.0, train_config.early_stop_min_delta))
        best_val = float("inf")
        bad_epochs = 0
        stopped_early = False

        pbar = tqdm(range(train_config.epochs), desc=desc, bar_format="{desc:<11.11}{percentage:3.0f}%|{bar:3}{r_bar}")
        for epoch_idx in pbar:
            self.model.train()
            self._train_step(dataloader_train, train_config, stage="train")
            self.model.eval()
            self._eval_step(dataloader_val, stage="val")
            self._eval_step(dataloader_test, stage="test")

            if scheduler_obj is not None:
                scheduler_obj.step()
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.history["lr"].append(current_lr)
            pbar.set_postfix_str(self._postfix(epoch_idx, current_lr if scheduler_obj is not None else None))

            if self.logger is not None:
                self.logger.log(
                    self._build_log_metrics(epoch_idx, current_lr if scheduler_obj is not None else None),
                    epoch_idx,
                )

            if early_stop_patience > 0:
                val_now = self.history["val"][epoch_idx]
                if val_now + early_stop_min_delta < best_val:
                    best_val = val_now
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        stopped_early = True
                        pbar.set_postfix_str(self._postfix(epoch_idx, current_lr) + " | early-stop")
                        break

        return {
            "history": self.history,
            "stopped_early": stopped_early,
            "epochs_completed": len(self.history["train"]),
        }
