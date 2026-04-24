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

        self.history = {"train": [], "test": [], "acc": []}

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
            acc_batches.append(acc_batch.detach())

        self.history[stage].append(fmean(loss_batches))
        self.history["acc"].append(fmean(acc_batches))

    def fit(self, dataloader_train, dataloader_test, lr, n_epochs=10, desc=None):

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        self.model.to(self.device)
        
        pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<11.11}{percentage:3.0f}%|{bar:3}{r_bar}")
        for epoch_idx in pbar:

            self.model.train() # train
            self._train_step(dataloader_train, stage="train")

            self.model.eval() # val/test
            self._eval_step(dataloader_test, stage="test")

            # print
            pbar.set_postfix_str("t={:.4f}, t*={:.4f}, ts={:.4f}, ts*={:.4f}, ts@t*={:.4f}, acc={:.4f}, acc*={:.4f}, acc@t*={:.4f}".format(
                self.history["train"][epoch_idx], # train
                min(self.history["train"][:epoch_idx+1]), # train*
                self.history["test"][epoch_idx], # test
                min(self.history["test"][:epoch_idx+1]), # test*
                self.history["test"][argmin(self.history["train"][:epoch_idx+1])], # test@train*
                self.history["acc"][epoch_idx], # acc
                max(self.history["acc"][:epoch_idx+1]), # acc*
                self.history["acc"][argmin(self.history["train"][:epoch_idx+1])], # acc@train*
            ))

            # log
            if self.logger is not None:
                self.logger.log({
                    "loss_train": self.history["train"][epoch_idx],
                    "loss_train_best": min(self.history["train"][:epoch_idx+1]),
                    "loss_test": self.history["test"][epoch_idx],
                    "loss_test_best": min(self.history["test"][:epoch_idx+1]),
                    "loss_test_at_val_best": self.history["test"][argmin(self.history["train"][:epoch_idx+1])],
                    "acc_test": self.history["acc"][epoch_idx],
                    "acc_test_best": max(self.history["acc"][:epoch_idx+1]),
                    "acc_test_at_val_best": self.history["acc"][argmin(self.history["train"][:epoch_idx+1])],
                },
                epoch_idx
            )
        
        # destruct logger
        # self.logger.end()


class TrainerPHTX(Trainer):

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
            acc_batches.append(acc_batch.detach())

        self.history[stage].append(fmean(loss_batches))
        self.history["acc"].append(fmean(acc_batches))