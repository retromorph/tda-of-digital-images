import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import load_config, run_experiment, safe_num_workers, seed_everything


def run_with_overrides(config_path: str, overrides: list[str]):
    cfg = load_config(config_path, overrides=overrides)
    cfg.num_workers = safe_num_workers(int(cfg.num_workers))
    seed_everything(int(cfg.seed))
    return run_experiment(cfg)


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotlist override, e.g. model.args.d_model=96 (can be passed multiple times).",
    )
    parser.add_argument("--inproc", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    run_with_overrides(args.config, args.override)


if __name__ == "__main__":
    main()
