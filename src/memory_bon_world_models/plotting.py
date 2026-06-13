from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "base_bon": "non-memory",
    "memory_bon": "naive memory",
    "oracle_memory_bon": "oracle memory",
    "repaired_memory_bon": "repaired memory",
}


def _main_slice(summary: pd.DataFrame, stale_fraction: float) -> pd.DataFrame:
    available = sorted(summary["stale_fraction"].unique())
    chosen = min(available, key=lambda value: abs(float(value) - stale_fraction))
    return summary[summary["stale_fraction"] == chosen].copy()


def plot_main_curves(summary: pd.DataFrame, output_dir: Path | str, *, stale_fraction: float = 0.85) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _main_slice(summary, stale_fraction)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    for strategy, group in data.groupby("strategy"):
        group = group.sort_values("n")
        label = LABELS.get(strategy, strategy)
        axes[0].plot(group["n"], group["true_return"], marker="o", label=label)
        axes[1].plot(group["n"], group["proxy_score"], marker="o", label=label)
    for axis, title, ylabel in [
        (axes[0], "True utility", "mean true return"),
        (axes[1], "Model proxy", "mean proxy score"),
    ]:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("candidate budget N")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / "candidate_budget_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    for strategy, group in data.groupby("strategy"):
        group = group.sort_values("n")
        label = LABELS.get(strategy, strategy)
        axes[0].plot(group["n"], group["retrieval_precision"], marker="o", label=label)
        axes[1].plot(group["n"], group["hallucinated_rollout"], marker="o", label=label)
    axes[0].set_ylabel("retrieval precision")
    axes[1].set_ylabel("hallucinated rollout rate")
    for axis, title in [(axes[0], "Selected retrieval"), (axes[1], "Rollout risk")]:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("candidate budget N")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / "retrieval_diagnostics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    max_n = int(data["n"].max())
    repair = data[data["n"] == max_n].copy()
    fig, axis = plt.subplots(figsize=(6.8, 3.6))
    order = ["base_bon", "memory_bon", "repaired_memory_bon", "oracle_memory_bon"]
    values = [float(repair[repair["strategy"] == strategy]["true_return"].iloc[0]) for strategy in order]
    colors = ["#4C78A8", "#D95F02", "#1B9E77", "#6A3D9A"]
    axis.bar([LABELS[strategy] for strategy in order], values, color=colors)
    axis.set_ylabel("mean true return at max N")
    axis.set_title("Repair closes much of the stale-memory gap")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / "repair_gap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(6.8, 3.6))
    n_max = int(summary["n"].max())
    subset = summary[(summary["n"] == n_max) & (summary["strategy"].isin(["memory_bon", "repaired_memory_bon"]))]
    for strategy, group in subset.groupby("strategy"):
        group = group.sort_values("stale_fraction")
        axis.plot(group["stale_fraction"], group["true_return"], marker="o", label=LABELS[strategy])
    axis.set_xlabel("stale memory fraction")
    axis.set_ylabel("mean true return at max N")
    axis.set_title("Staleness sensitivity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = output / "staleness_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    return paths
