from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .environment import REVERSE, RegimeWorld
from .memory import EpisodicMemory
from .models import BaseWorldModel, MemoryWorldModel, OracleMemoryWorldModel, RepairedMemoryWorldModel
from .planning import plan_best_of_n


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    trials: int = 120
    horizon: int = 8
    ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    stale_fractions: tuple[float, ...] = (0.45, 0.65, 0.85, 0.93)
    repair_penalty: float = 0.9
    memory_total: int = 320


def _models(memory: EpisodicMemory):
    return [
        BaseWorldModel(memory),
        MemoryWorldModel(memory),
        OracleMemoryWorldModel(memory),
        RepairedMemoryWorldModel(memory),
    ]


def run_suite(config: ExperimentConfig | None = None, *, output_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = ExperimentConfig()
    output = Path(output_dir) if output_dir is not None else Path("results") / "full"
    output.mkdir(parents=True, exist_ok=True)

    world = RegimeWorld(horizon=config.horizon)
    rows: list[dict[str, float | str]] = []
    root_rng = np.random.default_rng(config.seed)

    for stale_fraction in config.stale_fractions:
        for trial in range(config.trials):
            memory_seed = int(root_rng.integers(0, 2**31 - 1))
            candidate_seed = int(root_rng.integers(0, 2**31 - 1))
            memory_rng = np.random.default_rng(memory_seed)
            memory = EpisodicMemory.from_regime_mix(
                memory_rng,
                current_regime=REVERSE,
                stale_fraction=float(stale_fraction),
                total=config.memory_total,
                world=world,
            )
            for model in _models(memory):
                for n in config.ns:
                    # Common random candidates per trial/N across strategies.
                    local_seed = candidate_seed + int(n) * 9973
                    rng = np.random.default_rng(local_seed)
                    row = plan_best_of_n(
                        rng,
                        model,
                        world,
                        REVERSE,
                        n=int(n),
                        repair_penalty=config.repair_penalty,
                    )
                    row["trial"] = trial
                    row["seed"] = local_seed
                    row["stale_fraction"] = float(stale_fraction)
                    row["regime"] = REVERSE.name
                    rows.append(row)

    raw = pd.DataFrame(rows)
    group_cols = ["stale_fraction", "strategy", "n"]
    metric_cols = [
        "proxy_score",
        "predicted_final_error",
        "true_return",
        "true_final_error",
        "true_success",
        "proxy_true_gap",
        "hallucinated_rollout",
        "mean_selected_action",
        "retrieval_stale_rate",
        "retrieval_precision",
        "memory_gate",
        "retrieval_disagreement",
        "dropout_variance",
        "diagnostic_risk",
    ]
    summary = raw.groupby(group_cols, as_index=False)[metric_cols].mean()

    raw.to_csv(output / "raw_rollouts.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    return raw, summary


def smoke_config(seed: int = 7) -> ExperimentConfig:
    return ExperimentConfig(seed=seed, trials=18, ns=(1, 4, 16, 64), stale_fractions=(0.85,), memory_total=260)


def paper_config(seed: int = 7) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        trials=90,
        ns=(1, 2, 4, 8, 16, 32, 64),
        stale_fractions=(0.45, 0.65, 0.85, 0.93),
        memory_total=320,
    )
