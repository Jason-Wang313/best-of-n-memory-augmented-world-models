"""Exact Gymnasium toy-text transition-memory stress tests for v4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v4_toytext_memory"
PAPER = ROOT / "paper"
PAPER_FIG = PAPER / "figures"
N_VALUES = (1, 4, 16, 64)
STRATEGIES = ("recent_only", "naive_memory", "repaired_memory", "oracle_memory")


@dataclass(frozen=True)
class ToyTextSpec:
    name: str
    env_id: str
    horizon: int
    discount: float
    raw_scale: float = 1.0
    repair_risk: float = 4.0


TASKS = {
    "FrozenLake-v1": ToyTextSpec("FrozenLake-v1", "FrozenLake-v1", horizon=12, discount=0.97, raw_scale=4.0),
    "CliffWalkingSlippery-v1": ToyTextSpec(
        "CliffWalkingSlippery-v1",
        "CliffWalkingSlippery-v1",
        horizon=14,
        discount=0.99,
        raw_scale=1.0,
        repair_risk=5.0,
    ),
    "Taxi-v3": ToyTextSpec("Taxi-v3", "Taxi-v3", horizon=16, discount=0.97, raw_scale=1.0, repair_risk=3.5),
}


def _macro(name: str, value: str | int | float) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def make_env(env_id: str):
    if env_id == "FrozenLake-v1":
        return gym.make(env_id, is_slippery=True)
    return gym.make(env_id)


def seeded_start_state(env_id: str, seed: int) -> int:
    env = make_env(env_id)
    try:
        obs, _ = env.reset(seed=int(seed))
        return int(obs)
    finally:
        env.close()


@lru_cache(maxsize=8)
def transition_model(env_id: str):
    env = make_env(env_id)
    try:
        return env.unwrapped.P, int(env.action_space.n), int(env.observation_space.n)
    finally:
        env.close()


def expected_return(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> float:
    p_model, _, _ = transition_model(env_id)
    dist = {int(start_state): 1.0}
    total = 0.0
    for t, action in enumerate(actions):
        next_dist: dict[int, float] = {}
        for state, state_prob in dist.items():
            for prob, next_state, reward, terminated in p_model[int(state)][int(action)]:
                mass = float(state_prob) * float(prob)
                total += (float(discount) ** t) * mass * float(reward)
                if not terminated:
                    next_dist[int(next_state)] = next_dist.get(int(next_state), 0.0) + mass
        dist = next_dist
        if not dist:
            break
    return float(total)


def _row_col(env_id: str, state: int) -> tuple[int, int]:
    if env_id == "FrozenLake-v1":
        return divmod(int(state), 4)
    if env_id == "CliffWalkingSlippery-v1":
        return divmod(int(state), 12)
    raise ValueError(env_id)


@lru_cache(maxsize=2048)
def _taxi_decode(state: int) -> tuple[int, int, int, int]:
    env = make_env("Taxi-v3")
    try:
        return tuple(int(x) for x in env.unwrapped.decode(int(state)))  # type: ignore[return-value]
    finally:
        env.close()


@lru_cache(maxsize=2048)
def _potential(env_id: str, state: int) -> float:
    if env_id == "FrozenLake-v1":
        row, col = _row_col(env_id, state)
        return 1.0 - 0.08 * (abs(3 - row) + abs(3 - col))
    if env_id == "CliffWalkingSlippery-v1":
        row, col = _row_col(env_id, state)
        return -0.18 * (abs(3 - row) + abs(11 - col))
    if env_id == "Taxi-v3":
        taxi_row, taxi_col, passenger, dest = _taxi_decode(state)
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        if passenger < 4:
            target = locs[passenger]
            onboard_bonus = 0.0
        else:
            target = locs[dest]
            onboard_bonus = 5.0
        return onboard_bonus - 0.25 * (abs(target[0] - taxi_row) + abs(target[1] - taxi_col))
    raise ValueError(env_id)


def stale_memory_proxy(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> float:
    """Optimistic transition-memory score that underweights stale hazards."""
    p_model, _, _ = transition_model(env_id)
    state = int(start_state)
    total = 0.0
    action_list = [int(a) for a in actions]
    for t, action in enumerate(action_list):
        transitions = p_model[state][int(action)]

        def optimistic_value(item):
            _, next_state, reward, terminated = item
            adjusted = float(reward)
            if env_id == "CliffWalkingSlippery-v1" and adjusted <= -100.0:
                adjusted = -1.0
            if env_id == "Taxi-v3" and adjusted <= -10.0:
                adjusted = -0.7
            if env_id == "FrozenLake-v1" and bool(terminated) and adjusted <= 0.0:
                adjusted = -0.02
            return adjusted + 0.92 * _potential(env_id, int(next_state))

        prob, next_state, reward, terminated = max(transitions, key=optimistic_value)
        adjusted = float(reward)
        if env_id == "CliffWalkingSlippery-v1" and adjusted <= -100.0:
            adjusted = -1.0
        if env_id == "Taxi-v3" and adjusted <= -10.0:
            adjusted = -0.7
        if env_id == "FrozenLake-v1" and bool(terminated) and adjusted <= 0.0:
            adjusted = -0.02
        total += (float(discount) ** t) * adjusted
        state = int(next_state)
        if terminated:
            break
    total += (float(discount) ** len(action_list)) * _potential(env_id, state)
    return float(total)


def transition_diagnostics(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> dict[str, float]:
    action_list = [int(a) for a in actions]
    p_model, _, _ = transition_model(env_id)
    dist = {int(start_state): 1.0}
    entropy = 0.0
    catastrophe = 0.0
    illegal = 0.0
    for t, action in enumerate(action_list):
        next_dist: dict[int, float] = {}
        for state, state_prob in dist.items():
            transitions = p_model[int(state)][int(action)]
            probs = np.asarray([float(item[0]) for item in transitions], dtype=float)
            probs = probs[probs > 0]
            entropy += float(state_prob) * float(-np.sum(probs * np.log(probs))) / max(1, len(action_list))
            for prob, next_state, reward, terminated in transitions:
                mass = float(state_prob) * float(prob)
                reward = float(reward)
                if env_id == "FrozenLake-v1" and terminated and reward <= 0.0:
                    catastrophe += (float(discount) ** t) * mass
                if env_id == "CliffWalkingSlippery-v1" and reward <= -100.0:
                    catastrophe += (float(discount) ** t) * mass
                if env_id == "Taxi-v3" and reward <= -10.0:
                    illegal += (float(discount) ** t) * mass
                if not terminated:
                    next_dist[int(next_state)] = next_dist.get(int(next_state), 0.0) + mass
        dist = next_dist
        if not dist:
            break
    risk = float(catastrophe + 0.35 * illegal)
    return {"entropy": float(entropy), "catastrophe": float(catastrophe), "illegal": float(illegal), "risk": risk}


def sample_actions(env_id: str, rng: np.random.Generator, horizon: int) -> np.ndarray:
    if env_id == "FrozenLake-v1":
        actions = rng.choice(4, size=horizon, p=[0.08, 0.40, 0.44, 0.08])
        if rng.random() < 0.30:
            prefix = np.asarray([2, 2, 1, 1, 1, 2], dtype=int)
            actions[: min(horizon, len(prefix))] = prefix[: min(horizon, len(prefix))]
        return actions.astype(int)
    if env_id == "CliffWalkingSlippery-v1":
        actions = rng.choice(4, size=horizon, p=[0.32, 0.48, 0.04, 0.16])
        if rng.random() < 0.36:
            prefix = np.asarray([0] + [1] * max(0, horizon - 1), dtype=int)
            actions[: len(prefix)] = prefix[:horizon]
        return actions.astype(int)
    if env_id == "Taxi-v3":
        actions = rng.choice(6, size=horizon, p=[0.18, 0.18, 0.18, 0.18, 0.14, 0.14])
        if rng.random() < 0.45:
            spots = rng.choice(horizon, size=max(1, horizon // 4), replace=False)
            actions[spots] = rng.choice([4, 5], size=len(spots))
        return actions.astype(int)
    raise ValueError(env_id)


def build_candidates(spec: ToyTextSpec, seed: int, *, candidate_count: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    start = seeded_start_state(spec.env_id, seed)
    rows: list[dict[str, float | int | str]] = []
    for candidate_id in range(candidate_count):
        actions = sample_actions(spec.env_id, rng, spec.horizon)
        true = expected_return(spec.env_id, start, actions, spec.discount)
        stale = spec.raw_scale * stale_memory_proxy(spec.env_id, start, actions, spec.discount)
        diag = transition_diagnostics(spec.env_id, start, actions, spec.discount)
        repair = stale - spec.repair_risk * diag["risk"] - 0.30 * diag["entropy"]
        rows.append(
            {
                "benchmark": spec.name,
                "seed": int(seed),
                "candidate_id": int(candidate_id),
                "start_state": int(start),
                "true_return": float(true),
                "recent_only_proxy": float(true),
                "naive_memory_proxy": float(stale + 0.25 * max(0.0, stale - true)),
                "repaired_memory_proxy": float(repair),
                "oracle_memory_proxy": float(true),
                "risk": float(diag["risk"]),
                "entropy": float(diag["entropy"]),
                "catastrophe": float(diag["catastrophe"]),
                "illegal": float(diag["illegal"]),
            }
        )
    return pd.DataFrame(rows)


def selection_expectation(candidates: pd.DataFrame, strategy: str, n_value: int) -> dict[str, float | int | str]:
    score_col = f"{strategy}_proxy"
    ordered = candidates.sort_values(score_col, kind="mergesort").reset_index(drop=True)
    m = len(ordered)
    ranks = np.arange(1, m + 1, dtype=float)
    weights = (ranks / m) ** int(n_value) - ((ranks - 1) / m) ** int(n_value)
    weights = weights / weights.sum()
    return {
        "strategy": strategy,
        "N": int(n_value),
        "selected_true_return": float(np.dot(weights, ordered["true_return"].to_numpy(dtype=float))),
        "selected_proxy_score": float(np.dot(weights, ordered[score_col].to_numpy(dtype=float))),
        "selected_risk": float(np.dot(weights, ordered["risk"].to_numpy(dtype=float))),
        "selected_entropy": float(np.dot(weights, ordered["entropy"].to_numpy(dtype=float))),
    }


def _ci(values: list[float], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975))}


def summarize(curves: pd.DataFrame, effects: pd.DataFrame) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    mismatch_count = 0
    repair_count = 0
    risk_count = 0
    for benchmark in TASKS:
        local = effects[effects["benchmark"] == benchmark]
        proxy_ci = _ci(local["naive_proxy_delta_n64_vs_n1"].tolist(), seed=4100)
        true_ci = _ci(local["naive_true_delta_n64_vs_n1"].tolist(), seed=4200)
        repair_ci = _ci(local["repair_true_gain_n64_over_naive"].tolist(), seed=4300)
        risk_ci = _ci(local["naive_risk_delta_n64_vs_n1"].tolist(), seed=4400)
        mismatch = bool(proxy_ci["lo"] > 0.25 and true_ci["hi"] < 0.0)
        repair = bool(repair_ci["lo"] > 0.05)
        risk = bool(risk_ci["lo"] > 0.01)
        mismatch_count += int(mismatch)
        repair_count += int(repair)
        risk_count += int(risk)
        diagnostics[benchmark] = {
            "naive_proxy_delta_n64_vs_n1_ci": proxy_ci,
            "naive_true_delta_n64_vs_n1_ci": true_ci,
            "repair_true_gain_n64_over_naive_ci": repair_ci,
            "naive_risk_delta_n64_vs_n1_ci": risk_ci,
            "selected_tail_mismatch": mismatch,
            "repair_helped": repair,
            "risk_increased": risk,
        }
    return {
        "experiment": "v4_toytext_memory_benchmarks",
        "benchmarks": list(TASKS),
        "strategies": list(STRATEGIES),
        "n_values": list(N_VALUES),
        "curve_rows": int(len(curves)),
        "effect_rows": int(len(effects)),
        "key_result": {
            "selected_tail_mismatch_benchmark_count": int(mismatch_count),
            "repair_improvement_benchmark_count": int(repair_count),
            "naive_risk_increase_benchmark_count": int(risk_count),
        },
        "benchmark_diagnostics": diagnostics,
    }


def write_plot(curves: pd.DataFrame) -> Path:
    means = curves.groupby(["benchmark", "strategy", "N"], as_index=False)["selected_true_return"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    colors = {
        "recent_only": "#4C78A8",
        "naive_memory": "#D95F02",
        "repaired_memory": "#1B9E77",
        "oracle_memory": "#4B0082",
    }
    for axis, benchmark in zip(axes, TASKS):
        sub = means[means["benchmark"] == benchmark]
        for strategy in STRATEGIES:
            local = sub[sub["strategy"] == strategy].sort_values("N")
            axis.plot(local["N"], local["selected_true_return"], marker="o", color=colors[strategy], label=strategy.replace("_", " "))
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(N_VALUES))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_title(benchmark)
        axis.set_xlabel("candidate budget N")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("expected true utility")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    path = OUT / "v4_toytext_memory_benchmarks.pdf"
    fig.savefig(path)
    plt.close(fig)
    target = PAPER_FIG / path.name
    target.write_bytes(path.read_bytes())
    return path


def write_macros(payload: dict[str, object], curves: pd.DataFrame, effects: pd.DataFrame) -> None:
    key = payload["key_result"]  # type: ignore[index]
    max_n = max(N_VALUES)
    main = curves[curves["N"] == max_n]
    naive = main[main["strategy"] == "naive_memory"]["selected_true_return"].mean()
    repaired = main[main["strategy"] == "repaired_memory"]["selected_true_return"].mean()
    text = ""
    text += _macro("ToyTextBenchmarks", len(TASKS))
    text += _macro("ToyTextCurveRows", len(curves))
    text += _macro("ToyTextEffectRows", len(effects))
    text += _macro("ToyTextMismatchCount", key["selected_tail_mismatch_benchmark_count"])  # type: ignore[index]
    text += _macro("ToyTextRepairCount", key["repair_improvement_benchmark_count"])  # type: ignore[index]
    text += _macro("ToyTextRiskCount", key["naive_risk_increase_benchmark_count"])  # type: ignore[index]
    text += _macro("ToyTextNaiveMaxTrueMean", _fmt(naive, 2))
    text += _macro("ToyTextRepairedMaxTrueMean", _fmt(repaired, 2))
    text += _macro("ToyTextMeanRepairGain", _fmt(repaired - naive, 2))
    (PAPER / "v4_toytext_macros.tex").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [11, 17, 23, 29, 31, 37]
    candidate_frames: list[pd.DataFrame] = []
    curve_rows: list[dict[str, float | int | str]] = []
    effect_rows: list[dict[str, float | int | str]] = []
    for benchmark, spec in TASKS.items():
        for seed in seeds:
            candidates = build_candidates(spec, seed)
            candidate_frames.append(candidates)
            local_rows = []
            for strategy in STRATEGIES:
                for n_value in N_VALUES:
                    row = selection_expectation(candidates, strategy, n_value)
                    row.update({"benchmark": benchmark, "seed": int(seed)})
                    curve_rows.append(row)
                    local_rows.append(row)
            local = pd.DataFrame(local_rows)
            n1 = local[local["N"] == 1].set_index("strategy")
            n64 = local[local["N"] == 64].set_index("strategy")
            effect_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": int(seed),
                    "naive_proxy_delta_n64_vs_n1": float(
                        n64.loc["naive_memory", "selected_proxy_score"] - n1.loc["naive_memory", "selected_proxy_score"]
                    ),
                    "naive_true_delta_n64_vs_n1": float(
                        n64.loc["naive_memory", "selected_true_return"] - n1.loc["naive_memory", "selected_true_return"]
                    ),
                    "naive_risk_delta_n64_vs_n1": float(
                        n64.loc["naive_memory", "selected_risk"] - n1.loc["naive_memory", "selected_risk"]
                    ),
                    "repair_true_gain_n64_over_naive": float(
                        n64.loc["repaired_memory", "selected_true_return"] - n64.loc["naive_memory", "selected_true_return"]
                    ),
                }
            )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    curves = pd.DataFrame(curve_rows)
    effects = pd.DataFrame(effect_rows)
    candidates.to_csv(OUT / "candidate_records.csv", index=False)
    curves.to_csv(OUT / "selection_curves.csv", index=False)
    effects.to_csv(OUT / "benchmark_effects.csv", index=False)
    figure = write_plot(curves)
    payload = summarize(curves, effects)
    payload["artifacts"] = {
        "candidates": "results/v4_toytext_memory/candidate_records.csv",
        "curves": "results/v4_toytext_memory/selection_curves.csv",
        "effects": "results/v4_toytext_memory/benchmark_effects.csv",
        "figure": "paper/figures/v4_toytext_memory_benchmarks.pdf",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_macros(payload, curves, effects)
    print(f"v4 toy-text memory benchmarks complete: {OUT}")
    print(
        "benchmarks={benchmarks} mismatch={mismatch} repair={repair} risk={risk} figure={figure}".format(
            benchmarks=len(TASKS),
            mismatch=payload["key_result"]["selected_tail_mismatch_benchmark_count"],  # type: ignore[index]
            repair=payload["key_result"]["repair_improvement_benchmark_count"],  # type: ignore[index]
            risk=payload["key_result"]["naive_risk_increase_benchmark_count"],  # type: ignore[index]
            figure=figure,
        )
    )


if __name__ == "__main__":
    main()
