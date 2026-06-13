from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from memory_bon_world_models.experiment import ExperimentConfig, paper_config, run_suite, smoke_config, v2paper_config
from memory_bon_world_models.plotting import plot_main_curves


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic memory-impostor world-model experiments.")
    parser.add_argument("--preset", choices=["smoke", "v2paper", "paper", "custom"], default="paper")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "smoke":
        config = smoke_config(seed=args.seed)
        out = ROOT / "results" / "smoke"
    elif args.preset == "v2paper":
        config = v2paper_config(seed=args.seed)
        out = ROOT / "results" / "v2paper"
    elif args.preset == "paper":
        config = paper_config(seed=args.seed)
        out = ROOT / "results" / "paper"
    else:
        config = ExperimentConfig(seed=args.seed, trials=args.trials)
        out = ROOT / "results" / "custom"

    _, summary = run_suite(config, output_dir=out)
    figure_paths = plot_main_curves(summary, ROOT / "figures")
    paper_fig_dir = ROOT / "paper" / "figures"
    paper_fig_dir.mkdir(exist_ok=True)
    for path in figure_paths:
        shutil.copy2(path, paper_fig_dir / path.name)

    main = summary[(summary["stale_fraction"] == 0.85) & (summary["n"] == summary["n"].max())]
    compact = main[["strategy", "true_return", "proxy_score", "proxy_true_gap", "hallucinated_rollout"]]
    print(compact.to_string(index=False))
    print(f"wrote {out / 'summary.csv'}")
    print(f"wrote {len(figure_paths)} figures under {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
