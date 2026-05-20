import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
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
    eta_min: float = 0.0  # absolute lower bound for lr (PyTorch CosineAnnealingLR semantics)


@dataclass
class TrainConfig:
    epochs: int = 10              # outer loop cap; for non-epoch budgets this is epochs_hint
    budget_kind: str = "epochs"   # "epochs" | "steps" | "seconds"
    budget_value: int = 10        # target in the unit given by budget_kind
    batch_size: int = 128
    grad_accum: int = 1
    max_grad_norm: float = 0.0
    early_stop_patience: int = 0
    early_stop_metric: str = "loss_val"
    early_stop_min_delta: float = 0.0
    save_best_weights: bool = False
    best_weights_artifact_path: str = "weights/best.pt"
    # Wallclock-budget Pareto snapshots (seconds). Fires test eval at each boundary
    # and logs with fixed MLflow keys ``*_at_budget_{1..N}``. Requires budget_kind="seconds".
    # Pair with scheduler=none to avoid LR-schedule confounders.
    budget_snapshots: list[int] | None = None


class Trainer:
    """Single training loop driven entirely by `OptimConfig` + `TrainConfig`.

    Supports classification (CE + accuracy) and regression (MSE + RMSE/MAE/R^2),
    AdamW with optional warmup+cosine schedule, gradient accumulation, gradient
    clipping, and val-loss based early stopping.
    """

    def __init__(self, model, device, logger, task="classification", forward_takes_mask=False, target_stats=None):
        self.model = model
        self.device = device
        self.logger = logger
        self.task = task
        self.forward_takes_mask = forward_takes_mask
        self.target_stats = target_stats
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
        # Throughput tracking (samples/sec) per stage, with EMA smoothing.
        self._throughput_ema_alpha = 0.9
        self._throughput_ema = {"train": None, "eval": None}
        self._latest_throughput = {"train": None, "eval": None}

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

    def _eval_metrics(self, preds: list, targets: list) -> dict:
        """Aggregate metrics over the *whole* eval set.

        Concatenating all batches before computing R² is required: R² is not
        additive across batches, so averaging per-batch values is invalid and
        degenerates to 0 for small/size-1 eval batches (where within-batch
        target variance vanishes). RMSE/MAE/acc are computed on the full set
        too for consistency.
        """
        if not preds:
            if self.task == "regression":
                return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            return {"acc": float("nan")}
        y_hat = torch.cat(preds, dim=0)
        y = torch.cat(targets, dim=0)
        if self.task == "regression":
            return self._regression_metrics(y_hat, y)
        return self._classification_metrics(y_hat, y)

    def _regression_metrics(self, y_hat, y):
        y_hat, y = self._prepare_regression_targets(y_hat, y)
        diff = y_hat - y
        # R² is scale-invariant so we compute it on whichever space we already have.
        ss_res = torch.sum(diff * diff)
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) * (y - y_mean))
        r2 = 1.0 - (ss_res / ss_tot) if float(ss_tot) > 0.0 else torch.tensor(0.0, device=y.device)
        # Report RMSE/MAE in original target units when standardization was applied.
        scale = float(self.target_stats["std"]) if self.target_stats else 1.0
        scaled_diff = diff * scale
        mse = torch.mean(scaled_diff * scaled_diff)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(scaled_diff))
        return {
            "rmse": float(rmse.detach().cpu().item()),
            "mae": float(mae.detach().cpu().item()),
            "r2": float(r2.detach().cpu().item()),
        }

    def _update_throughput(self, stage_key: str, samples: int, elapsed_s: float) -> None:
        """Record per-epoch throughput (samples/sec) and update its EMA."""
        if samples <= 0 or elapsed_s <= 0:
            return
        latest = samples / elapsed_s
        self._latest_throughput[stage_key] = latest
        prev = self._throughput_ema[stage_key]
        if prev is None:
            self._throughput_ema[stage_key] = latest
        else:
            alpha = self._throughput_ema_alpha
            self._throughput_ema[stage_key] = alpha * prev + (1.0 - alpha) * latest

    def _train_step(self, dataloader, train_cfg: TrainConfig, stage):
        loss_batches = []
        accum = max(1, int(train_cfg.grad_accum))
        max_norm = float(train_cfg.max_grad_norm)

        def _flush():
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1
            if self._budget_kind == "seconds":
                self._maybe_fire_snapshots()
            if self._budget_kind == "steps" and self._global_step >= self._budget_value:
                self._budget_reached = True
            elif self._budget_kind == "seconds" and (time.monotonic() - self._budget_started_at) >= self._budget_value:
                self._budget_reached = True

        samples_seen = 0
        epoch_start = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(dataloader):
            x, mask, y = self._unpack_batch(batch)
            samples_seen += int(x.shape[0])
            y_hat = self._forward(x, mask)
            if self.task == "regression":
                y_hat, y = self._prepare_regression_targets(y_hat, y)
            loss = self.loss_fn(y_hat, y)
            (loss / accum).backward()
            loss_batches.append(float(loss.detach().cpu().item()))
            if (step_idx + 1) % accum == 0:
                _flush()
                if self._budget_reached:
                    break
        # flush leftover grads only if we didn't stop mid-epoch
        if not self._budget_reached and (len(loss_batches) % accum) != 0:
            _flush()
        self._update_throughput("train", samples_seen, time.monotonic() - epoch_start)
        self.history[stage].append(fmean(loss_batches) if loss_batches else float("nan"))

    def _eval_pass(self, dataloader, *, with_metrics: bool) -> dict:
        """Side-effect-free eval pass; returns metrics dict, does NOT touch history."""
        loss_batches: list[float] = []
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, mask, y = self._unpack_batch(batch)
                y_hat = self._forward(x, mask)
                if self.task == "regression":
                    y_hat, y = self._prepare_regression_targets(y_hat, y)
                loss = self.loss_fn(y_hat, y)
                loss_batches.append(float(loss.detach().cpu().item()))
                if with_metrics:
                    preds.append(y_hat.detach())
                    targets.append(y.detach())
        if was_training:
            self.model.train()
        out: dict = {"loss": fmean(loss_batches) if loss_batches else float("nan")}
        if with_metrics:
            out.update(self._eval_metrics(preds, targets))
        return out

    def _maybe_fire_snapshots(self) -> None:
        """If the current wallclock has crossed any pending snapshot boundary, run eval and log."""
        if not self._snapshot_targets:
            return
        elapsed = time.monotonic() - self._budget_started_at
        while self._snapshot_next_idx < len(self._snapshot_targets) and elapsed >= self._snapshot_targets[self._snapshot_next_idx]:
            slot = self._snapshot_next_idx + 1  # 1-indexed for MLflow keys
            val_metrics = self._eval_pass(self._snapshot_val_loader, with_metrics=False)
            test_metrics = self._eval_pass(self._snapshot_test_loader, with_metrics=True)
            payload = {
                f"loss_val_at_budget_{slot}": val_metrics["loss"],
                f"loss_test_at_budget_{slot}": test_metrics["loss"],
                f"elapsed_at_budget_{slot}": elapsed,
                f"steps_at_budget_{slot}": float(self._global_step),
            }
            if self.task == "regression":
                payload.update(
                    {
                        f"rmse_at_budget_{slot}": test_metrics["rmse"],
                        f"mae_at_budget_{slot}": test_metrics["mae"],
                        f"r2_at_budget_{slot}": test_metrics["r2"],
                    }
                )
            else:
                payload.update({f"acc_at_budget_{slot}": test_metrics["acc"]})
            if self.logger is not None:
                self.logger.log(payload, step=self._global_step)
            self._snapshot_next_idx += 1

    def _eval_step(self, dataloader, stage):
        loss_batches = []
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        samples_seen = 0
        epoch_start = time.monotonic()
        with torch.no_grad():
            for batch in dataloader:
                x, mask, y = self._unpack_batch(batch)
                samples_seen += int(x.shape[0])
                y_hat = self._forward(x, mask)
                if self.task == "regression":
                    y_hat, y = self._prepare_regression_targets(y_hat, y)
                loss = self.loss_fn(y_hat, y)
                loss_batches.append(float(loss.detach().cpu().item()))

                if stage == "test":
                    preds.append(y_hat.detach())
                    targets.append(y.detach())

        # Track eval throughput on the val pass (test repeats the same compute).
        if stage == "val":
            self._update_throughput("eval", samples_seen, time.monotonic() - epoch_start)
        self.history[stage].append(fmean(loss_batches) if loss_batches else float("nan"))
        if stage == "test":
            metrics = self._eval_metrics(preds, targets)
            if self.task == "regression":
                for key in ("rmse", "mae", "r2"):
                    self.history[key].append(metrics[key])
            else:
                self.history["acc"].append(metrics["acc"])

    def _build_optimizer(self, cfg: OptimConfig):
        if cfg.optimizer.lower() != "adamw":
            raise ValueError("Only AdamW optimizer is supported (got '{}').".format(cfg.optimizer))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _build_scheduler(self, cfg: OptimConfig, n_epochs: int):
        scheduler_name = cfg.scheduler.lower()
        if scheduler_name == "none":
            return None
        eta_min_abs = float(cfg.eta_min)
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
        if self._latest_throughput["train"] is not None:
            payload["samples_per_s_train"] = self._latest_throughput["train"]
            payload["samples_per_s_train_ema"] = self._throughput_ema["train"]
        if self._latest_throughput["eval"] is not None:
            payload["samples_per_s_eval"] = self._latest_throughput["eval"]
            payload["samples_per_s_eval_ema"] = self._throughput_ema["eval"]
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
        self._budget_kind = train_config.budget_kind
        self._budget_value = train_config.budget_value
        self._budget_started_at = time.monotonic()
        self._global_step = 0
        self._budget_reached = False

        # Snapshot state for Pareto-curve logging at wallclock boundaries.
        snapshots = train_config.budget_snapshots or []
        if snapshots and train_config.budget_kind != "seconds":
            raise ValueError("budget_snapshots requires budget.kind='seconds'.")
        self._snapshot_targets = sorted(int(s) for s in snapshots)
        for s in self._snapshot_targets:
            if s > train_config.budget_value:
                raise ValueError(
                    f"budget_snapshot {s}s exceeds budget.value={train_config.budget_value}s; "
                    "all snapshots must fire before the run ends."
                )
        self._snapshot_next_idx = 0
        self._snapshot_val_loader = dataloader_val
        self._snapshot_test_loader = dataloader_test
        if self._snapshot_targets and self.logger is not None:
            self.logger.log(
                {f"budget_seconds_{i + 1}": float(s) for i, s in enumerate(self._snapshot_targets)},
                step=0,
            )

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
        best_state = None
        best_epoch = -1

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

            val_now = self.history["val"][epoch_idx]
            improved = val_now + early_stop_min_delta < best_val
            if improved:
                best_val = val_now
                bad_epochs = 0
                if train_config.save_best_weights:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch_idx
            elif early_stop_patience > 0:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    stopped_early = True
                    pbar.set_postfix_str(self._postfix(epoch_idx, current_lr) + " | early-stop")
                    break

            if self._budget_reached:
                pbar.set_postfix_str(self._postfix(epoch_idx, current_lr) + " | budget")
                break

        epochs_completed = len(self.history["train"])
        final_step = max(0, epochs_completed - 1)
        if (
            train_config.save_best_weights
            and self.logger is not None
            and best_state is not None
        ):
            rel = Path(train_config.best_weights_artifact_path)
            artifact_dir = None if rel.parent == Path(".") else str(rel.parent)
            payload = {
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val),
            }
            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td) / rel.name
                torch.save(payload, str(tmp_path))
                self.logger.log_artifact(str(tmp_path), artifact_path=artifact_dir)
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            self.logger.log(
                {"best_val_loss": float(best_val), "best_epoch": float(best_epoch)},
                step=final_step,
            )

        return {
            "history": self.history,
            "stopped_early": stopped_early,
            "stopped_by_budget": self._budget_reached,
            "epochs_completed": len(self.history["train"]),
            "total_steps": self._global_step,
        }
