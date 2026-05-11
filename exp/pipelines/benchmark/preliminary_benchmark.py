import argparse
from pathlib import Path
import warnings

from exp.pipelines.orchestrator import run

ROOT = Path(__file__).resolve().parents[3]


warnings.warn(
    "exp/pipelines/benchmark/preliminary_benchmark.py is deprecated; forwarding to exp/pipelines/orchestrator.py",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper for unified orchestrator."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "exp" / "config" / "benchmark" / "preliminary_clean.yaml"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_method", default=None, help="Run only one method key from config.")
    parser.add_argument("--only_task", default=None, help="Run only one task name from config.")
    args = parser.parse_args()

    raise SystemExit(
        run(
            cfg_path=Path(args.config),
            dry_run=args.dry_run,
            only_method=args.only_method,
            only_task=args.only_task,
            inproc=False,
        )
    )
