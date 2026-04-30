import os

import torch
import torch.nn as nn

from datetime import datetime
try:
   from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at runtime
   load_dotenv = None

if load_dotenv is not None:
   # Keep explicit shell env vars higher priority than .env values.
   load_dotenv(override=False)


def get_mlflow_tracking_uri() -> str:
    """MLflow tracking URI from env/.env; default local ``sqlite:///mlruns/mlflow.db``."""
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")


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
