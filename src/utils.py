import os

import torch
import torch.nn as nn

from datetime import datetime


def get_mlflow_tracking_uri() -> str:
    """MLflow tracking URI: set ``MLFLOW_TRACKING_URI``; default local ``file:./mlruns``."""
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


def get_activation(activation_str, alpha=0.0):
   activations = {
      "GELU": nn.GELU(),
      "ELU": nn.ELU(),
      "ReLU": nn.ReLU(),
      "LeakyReLU": nn.LeakyReLU(alpha),
      "CELU": nn.CELU(alpha)
   }
   
   if activation_str not in activations.keys():
      raise ValueError("Supported activations are '{}'.".format("', '".join(activations.keys())))

   return activations[activation_str]


def argmin(lst):
  return lst.index(min(lst))


def save_checkpoint(model, optimizer, args, dir="./chk"):
   torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args_dict": vars(args),
        "args": args,
        "datetime": datetime.now()
    }, "{}/{}_{}_{}_seed-{}_epochs-{}.pt".format(dir, args.model, args.dataset, args.experiment, args.seed, args.epochs))
