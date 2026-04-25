import torch

from torcheval.metrics.functional import multiclass_accuracy
from statistics import fmean
from .utils import argmin
from tqdm import tqdm


class Trainer():

    def __init__(self, model, device, logger):
        self.model = model
        self.device = device
        self.logger = logger

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_fn = multiclass_accuracy

        self.history = {"train": [], "val": [], "test": [], "acc": []}

    def _train_step(self, dataloader, stage):

        loss_batches = []
        for X, Y in dataloader:
            X, Y = X.to(self.device), Y.to(self.device)
            
            Y_hat = self.model(X)

            loss_batch = self.loss_fn(Y_hat, Y)
            
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                loss_batches.append(loss_batch.detach())

        self.history[stage].append(fmean(loss_batches))

    def _eval_step(self, dataloader, stage):

        loss_batches = []
        acc_batches = []
        for X, Y in dataloader:
            X, Y = X.to(self.device), Y.to(self.device)
            
            Y_hat = self.model(X)
            loss_batch = self.loss_fn(Y_hat, Y)
            acc_batch = self.metric_fn(Y_hat, Y).detach()

            loss_batches.append(loss_batch.detach())
            if stage=="test":
                acc_batches.append(acc_batch.detach())

        self.history[stage].append(fmean(loss_batches))
        if stage=="test":
            self.history["acc"].append(fmean(acc_batches))

    def fit(self, dataloader_train, dataloader_val, dataloader_test, lr, n_epochs=10, desc=None):

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        
        pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<11.11}{percentage:3.0f}%|{bar:3}{r_bar}")
        for epoch_idx in pbar:

            self.model.train() # train
            self._train_step(dataloader_train, stage="train")

            self.model.eval() # val/test
            self._eval_step(dataloader_val, stage="val")
            self._eval_step(dataloader_test, stage="test")

            # print
            pbar.set_postfix_str("t={:.4f}, t*={:.4f}, v={:.4f}, v*={:.4f}, ts={:.4f}, ts*={:.4f}, ts@t*={:.4f}, ts@v*={:.4f}, acc={:.4f}, acc*={:.4f}, acc@t*={:.4f}, acc@v*={:.4f}".format(
                self.history["train"][epoch_idx], # train
                min(self.history["train"][:epoch_idx+1]), # train*
                
                self.history["val"][epoch_idx], # val
                min(self.history["val"][:epoch_idx+1]), # val*
                
                self.history["test"][epoch_idx], # test
                min(self.history["test"][:epoch_idx+1]), # test*
                self.history["test"][argmin(self.history["train"][:epoch_idx+1])], # test@train*
                self.history["test"][argmin(self.history["val"][:epoch_idx+1])], # test@val*
                
                self.history["acc"][epoch_idx], # acc
                max(self.history["acc"][:epoch_idx+1]), # acc*
                self.history["acc"][argmin(self.history["train"][:epoch_idx+1])], # acc@train*
                self.history["acc"][argmin(self.history["val"][:epoch_idx+1])], # acc@val*
            ))

            # log
            if self.logger is not None:
                self.logger.log({
                    "loss_train": self.history["train"][epoch_idx],
                    "loss_train_best": min(self.history["train"][:epoch_idx+1]),

                    "loss_val": self.history["val"][epoch_idx],
                    "loss_val_best": min(self.history["val"][:epoch_idx+1]),

                    "loss_test": self.history["test"][epoch_idx],
                    "loss_test_best": min(self.history["test"][:epoch_idx+1]),
                    "loss_test_at_train_best": self.history["test"][argmin(self.history["train"][:epoch_idx+1])],
                    "loss_test_at_val_best": self.history["test"][argmin(self.history["val"][:epoch_idx+1])],
                    
                    "acc_test": self.history["acc"][epoch_idx],
                    "acc_test_best": max(self.history["acc"][:epoch_idx+1]),
                    "acc_test_at_train_best": self.history["acc"][argmin(self.history["train"][:epoch_idx+1])],
                    "acc_test_at_val_best": self.history["acc"][argmin(self.history["val"][:epoch_idx+1])],
                },
                epoch_idx
            )
        
        # destruct logger
        if self.logger is not None:
            self.logger.end()


class TrainerPersformer(Trainer):

    def _train_step(self, dataloader, stage):

        loss_batches = []
        for X, mask, Y in dataloader:
            X, mask, Y = X.to(self.device), mask.to(self.device), Y.to(self.device)
            
            Y_hat = self.model(X, mask)
            loss_batch = self.loss_fn(Y_hat, Y)
            
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                loss_batches.append(loss_batch.detach())

        self.history[stage].append(fmean(loss_batches))

    def _eval_step(self, dataloader, stage):

        loss_batches = []
        acc_batches = []
        for X, mask, Y in dataloader:
            X, mask, Y = X.to(self.device), mask.to(self.device), Y.to(self.device)
            
            Y_hat = self.model(X, mask)
            loss_batch = self.loss_fn(Y_hat, Y)
            acc_batch = self.metric_fn(Y_hat, Y).detach()

            loss_batches.append(loss_batch.detach())
            if stage == "test":
                acc_batches.append(acc_batch.detach())

        self.history[stage].append(fmean(loss_batches))
        if stage == "test":
            self.history["acc"].append(fmean(acc_batches))

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        lr,
        n_epochs=10,
        desc=None,
        *,
        weight_decay=0.0,
        warmup_epochs=0,
        eta_min=0.0,
    ):
        """AdamW + linear warmup + cosine decay (Persformer path).

        ``eta_min`` is ``LR_min / LR_peak`` (ratio), matching ``run_persformer.py``.
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.model.to(self.device)
        # ratio -> absolute floor for torch cosine scheduler
        eta_min_abs = lr * float(eta_min)
        warmup_epochs = int(max(0, warmup_epochs))

        if warmup_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, n_epochs),
                eta_min=eta_min_abs,
            )
        elif warmup_epochs >= n_epochs:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0 / warmup_epochs,
                end_factor=1.0,
                total_iters=max(1, n_epochs),
            )
        else:
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
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        if warmup_epochs > 0:
            initial_lr = lr * (1.0 / warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = initial_lr

        pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<11.11}{percentage:3.0f}%|{bar:3}{r_bar}")
        for epoch_idx in pbar:
            self.model.train()
            self._train_step(dataloader_train, stage="train")

            self.model.eval()
            self._eval_step(dataloader_val, stage="val")
            self._eval_step(dataloader_test, stage="test")
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix_str(
                "t={:.4f}, t*={:.4f}, v={:.4f}, v*={:.4f}, ts={:.4f}, ts*={:.4f}, ts@t*={:.4f}, ts@v*={:.4f}, acc={:.4f}, acc*={:.4f}, acc@t*={:.4f}, acc@v*={:.4f}, lr={:.2e}".format(
                    self.history["train"][epoch_idx],
                    min(self.history["train"][: epoch_idx + 1]),
                    self.history["val"][epoch_idx],
                    min(self.history["val"][: epoch_idx + 1]),
                    self.history["test"][epoch_idx],
                    min(self.history["test"][: epoch_idx + 1]),
                    self.history["test"][argmin(self.history["train"][: epoch_idx + 1])],
                    self.history["test"][argmin(self.history["val"][: epoch_idx + 1])],
                    self.history["acc"][epoch_idx],
                    max(self.history["acc"][: epoch_idx + 1]),
                    self.history["acc"][argmin(self.history["train"][: epoch_idx + 1])],
                    self.history["acc"][argmin(self.history["val"][: epoch_idx + 1])],
                    current_lr,
                )
            )

            if self.logger is not None:
                self.logger.log(
                    {
                        "loss_train": self.history["train"][epoch_idx],
                        "loss_train_best": min(self.history["train"][: epoch_idx + 1]),
                        "loss_val": self.history["val"][epoch_idx],
                        "loss_val_best": min(self.history["val"][: epoch_idx + 1]),
                        "loss_test": self.history["test"][epoch_idx],
                        "loss_test_best": min(self.history["test"][: epoch_idx + 1]),
                        "loss_test_at_train_best": self.history["test"][
                            argmin(self.history["train"][: epoch_idx + 1])
                        ],
                        "loss_test_at_val_best": self.history["test"][
                            argmin(self.history["val"][: epoch_idx + 1])
                        ],
                        "acc_test": self.history["acc"][epoch_idx],
                        "acc_test_best": max(self.history["acc"][: epoch_idx + 1]),
                        "acc_test_at_train_best": self.history["acc"][
                            argmin(self.history["train"][: epoch_idx + 1])
                        ],
                        "acc_test_at_val_best": self.history["acc"][
                            argmin(self.history["val"][: epoch_idx + 1])
                        ],
                        "lr": current_lr,
                    },
                    epoch_idx,
                )

        if self.logger is not None:
            self.logger.end()
