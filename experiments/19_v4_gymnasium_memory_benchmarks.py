"""Run lightweight Gymnasium memory-impostor benchmark cards for v4.

The experiment is intentionally CPU-scale. It uses standard Gymnasium
classic-control environments as executable transfer cards, then injects stale
retrieval memory by storing transitions whose action labels come from the
current query but whose next-state deltas come from the mirrored action. The
naive scorer treats dense stale support as confidence; the repaired scorer
penalizes the same support-risk signal. The result is not a SOTA RL claim; it
is a standard-environment stress test for whether stale memory can make the
selected high-score rollout less executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v4_gymnasium_memory"
PAPER = ROOT / "paper"
PAPER_FIG = PAPER / "figures"
N_VALUES = (1, 4, 16, 64)
STRATEGIES = ("recent_only", "naive_memory", "repaired_memory", "oracle_memory")


@dataclass(frozen=True)
class ClassicSpec:
    name: str
    env_id: str
    horizon: int
    discount: float
    actions: tuple[int, ...]
    state_scale: tuple[float, ...]


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    regime: str
    timestamp: float

    @property
    def delta(self) -> np.ndarray:
        return self.next_state - self.state

    @property
    def is_recent(self) -> bool:
        return self.timestamp >= 1000.0


TASKS = {
    "CartPole-v1": ClassicSpec(
        name="CartPole-v1",
        env_id="CartPole-v1",
        horizon=20,
        discount=0.99,
        actions=(0, 1),
        state_scale=(2.4, 2.5, 0.2095, 2.5),
    ),
    "MountainCar-v0": ClassicSpec(
        name="MountainCar-v0",
        env_id="MountainCar-v0",
        horizon=25,
        discount=0.99,
        actions=(0, 1, 2),
        state_scale=(0.6, 0.07),
    ),
    "Acrobot-v1": ClassicSpec(
        name="Acrobot-v1",
        env_id="Acrobot-v1",
        horizon=25,
        discount=0.99,
        actions=(0, 1, 2),
        state_scale=(np.pi, np.pi, 4 * np.pi, 9 * np.pi),
    ),
}


def _macro(name: str, value: str | int | float) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def mirror_action(spec: ClassicSpec, action: int) -> int:
    if len(spec.actions) == 2:
        return 1 - int(action)
    if int(action) == 0:
        return 2
    if int(action) == 2:
        return 0
    return 1


def start_state(spec: ClassicSpec, seed: int) -> np.ndarray:
    env = gym.make(spec.env_id)
    try:
        env.reset(seed=int(seed))
        return np.asarray(env.unwrapped.state, dtype=float).copy()
    finally:
        env.close()


def _set_state(env, state: np.ndarray) -> None:
    env.unwrapped.state = np.asarray(state, dtype=float).copy()


def _step_from_state(env, state: np.ndarray, action: int) -> np.ndarray:
    _set_state(env, state)
    env.step(int(action))
    return np.asarray(env.unwrapped.state, dtype=float).copy()


def _sample_memory_state(spec: ClassicSpec, rng: np.random.Generator, anchor: np.ndarray) -> np.ndarray:
    if spec.name == "CartPole-v1":
        noise = rng.normal(0.0, [0.30, 0.45, 0.055, 0.45])
        state = np.asarray(anchor, dtype=float) + noise
        state[0] = np.clip(state[0], -1.6, 1.6)
        state[2] = np.clip(state[2], -0.16, 0.16)
        return state
    if spec.name == "MountainCar-v0":
        return np.asarray([rng.uniform(-1.12, 0.28), rng.uniform(-0.055, 0.055)], dtype=float)
    if spec.name == "Acrobot-v1":
        return np.asarray(
            [
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-2.5, 2.5),
                rng.uniform(-4.0, 4.0),
            ],
            dtype=float,
        )
    raise ValueError(spec.name)


def build_memory(spec: ClassicSpec, seed: int, *, stale_n: int = 60, recent_n: int = 30) -> list[Transition]:
    rng = np.random.default_rng(int(seed))
    anchor = start_state(spec, seed)
    env = gym.make(spec.env_id)
    transitions: list[Transition] = []
    try:
        env.reset(seed=int(seed))
        for _ in range(stale_n):
            state = _sample_memory_state(spec, rng, anchor)
            action = int(rng.choice(spec.actions))
            next_state = _step_from_state(env, state, mirror_action(spec, action))
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    regime="stale_mirrored",
                    timestamp=float(rng.uniform(0.0, 100.0)),
                )
            )
        for _ in range(recent_n):
            state = _sample_memory_state(spec, rng, anchor)
            action = int(rng.choice(spec.actions))
            next_state = _step_from_state(env, state, action)
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    regime="current",
                    timestamp=float(rng.uniform(1000.0, 1100.0)),
                )
            )
    finally:
        env.close()
    rng.shuffle(transitions)
    return transitions


def _cartpole_margin(state: np.ndarray) -> float:
    x, _, theta, _ = np.asarray(state, dtype=float)
    x_margin = max(0.0, 1.0 - abs(float(x)) / 2.4)
    theta_margin = max(0.0, 1.0 - abs(float(theta)) / 0.2095)
    return 0.5 * x_margin + 0.5 * theta_margin


def _acrobot_tip_height(state: np.ndarray) -> float:
    theta1, theta2 = float(state[0]), float(state[1])
    return -np.cos(theta1) - np.cos(theta1 + theta2)


def shaped_utility(spec: ClassicSpec, states: list[np.ndarray], actions: Iterable[int]) -> float:
    action_list = list(actions)
    final_state = states[-1] if states else np.zeros(len(spec.state_scale), dtype=float)
    if spec.name == "CartPole-v1":
        alive = 0.0
        for t, state in enumerate(states):
            x, _, theta, _ = np.asarray(state, dtype=float)
            if abs(float(x)) > 2.4 or abs(float(theta)) > 0.2095:
                break
            alive += spec.discount**t
        return float(alive + 8.0 * _cartpole_margin(final_state))
    if spec.name == "MountainCar-v0":
        max_position = max(float(s[0]) for s in states) if states else float(final_state[0])
        final_position = float(final_state[0])
        final_velocity = float(final_state[1])
        return float(35.0 * (max_position + 0.5) + 15.0 * max(0.0, final_position + 0.5) + 8.0 * final_velocity)
    if spec.name == "Acrobot-v1":
        heights = [_acrobot_tip_height(state) for state in states] or [_acrobot_tip_height(final_state)]
        return float(14.0 * max(heights) + 8.0 * _acrobot_tip_height(final_state))
    raise ValueError(spec.name)


def execute_true(spec: ClassicSpec, initial_state: np.ndarray, actions: Iterable[int]) -> tuple[float, dict[str, float]]:
    env = gym.make(spec.env_id)
    states: list[np.ndarray] = []
    terminated_at = int(spec.horizon)
    try:
        env.reset(seed=0)
        _set_state(env, initial_state)
        for t, action in enumerate(actions):
            _, _, terminated, truncated, _ = env.step(int(action))
            states.append(np.asarray(env.unwrapped.state, dtype=float).copy())
            if terminated or truncated:
                terminated_at = t + 1
                break
    finally:
        env.close()
    value = shaped_utility(spec, states, actions)
    return value, {"terminated_at": float(terminated_at), "final_state_norm": float(np.linalg.norm(states[-1])) if states else 0.0}


def sample_actions(spec: ClassicSpec, rng: np.random.Generator) -> np.ndarray:
    horizon = int(spec.horizon)
    if spec.name == "CartPole-v1":
        if rng.random() < 0.34:
            return np.full(horizon, int(rng.choice(spec.actions)), dtype=int)
        if rng.random() < 0.55:
            switch = int(rng.integers(max(2, horizon // 5), max(3, 4 * horizon // 5)))
            first = int(rng.choice(spec.actions))
            out = np.full(horizon, first, dtype=int)
            out[switch:] = 1 - first
            return out
        return rng.choice(spec.actions, size=horizon, p=[0.50, 0.50]).astype(int)
    if spec.name == "MountainCar-v0":
        if rng.random() < 0.30:
            return np.full(horizon, 2, dtype=int)
        if rng.random() < 0.52:
            period = int(rng.integers(6, 14))
            phase = int(rng.integers(0, period))
            return np.asarray([0 if ((t + phase) % period) < period // 2 else 2 for t in range(horizon)], dtype=int)
        return rng.choice(spec.actions, size=horizon, p=[0.32, 0.10, 0.58]).astype(int)
    if spec.name == "Acrobot-v1":
        if rng.random() < 0.34:
            return np.full(horizon, int(rng.choice([0, 2])), dtype=int)
        if rng.random() < 0.55:
            period = int(rng.integers(5, 12))
            return np.asarray([0 if (t % period) < period // 2 else 2 for t in range(horizon)], dtype=int)
        return rng.choice(spec.actions, size=horizon, p=[0.42, 0.10, 0.48]).astype(int)
    raise ValueError(spec.name)


class VectorMemoryModel:
    def __init__(self, spec: ClassicSpec, transitions: list[Transition], strategy: str, *, k: int = 7):
        self.spec = spec
        self.transitions = list(transitions)
        self.strategy = strategy
        self.k = int(k)
        self.states = np.vstack([item.state for item in transitions])
        self.actions = np.asarray([item.action for item in transitions], dtype=int)
        self.next_states = np.vstack([item.next_state for item in transitions])
        self.deltas = self.next_states - self.states
        self.is_recent = np.asarray([item.is_recent for item in transitions], dtype=bool)
        self.scale = np.asarray(spec.state_scale, dtype=float)

    def _retrieve(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        state_diff = (self.states - np.asarray(state, dtype=float)) / self.scale
        dist = np.sum(state_diff * state_diff, axis=1)
        dist = dist + np.where(self.actions == int(action), 0.0, 2.25)
        if self.strategy in {"recent_only", "repaired_memory"}:
            dist = dist + np.where(self.is_recent, 0.0, 1.65)
        elif self.strategy == "oracle_memory":
            dist = dist + np.where(self.is_recent, 0.0, 1e6)
        elif self.strategy != "naive_memory":
            raise ValueError(self.strategy)
        k = min(self.k, len(dist))
        idx = np.argpartition(dist, k - 1)[:k]
        idx = idx[np.argsort(dist[idx])]
        weights = 1.0 / (np.sqrt(dist[idx]) + 1e-3)
        weights = weights / np.sum(weights)
        return idx, weights

    def predict_next(self, state: np.ndarray, action: int) -> tuple[np.ndarray, dict[str, float]]:
        idx, weights = self._retrieve(state, action)
        deltas = self.deltas[idx]
        mean_delta = np.average(deltas, axis=0, weights=weights)
        scaled = deltas / self.scale
        scaled_mean = mean_delta / self.scale
        spread = float(np.average(np.linalg.norm(scaled - scaled_mean, axis=1), weights=weights))
        stale_rate = float(np.mean(~self.is_recent[idx]))
        precision = float(np.mean(self.is_recent[idx]))
        if len(idx) > 2:
            loo = []
            for i in range(len(idx)):
                mask = np.ones(len(idx), dtype=bool)
                mask[i] = False
                local_w = weights[mask] / np.sum(weights[mask])
                loo.append(np.average(deltas[mask] / self.scale, axis=0, weights=local_w))
            dropout = float(np.mean(np.var(np.vstack(loo), axis=0)))
        else:
            dropout = 0.0
        risk = float(stale_rate + 0.45 * spread + min(1.5, 8.0 * dropout))
        return np.asarray(state, dtype=float) + mean_delta, {
            "stale_rate": stale_rate,
            "precision": precision,
            "spread": spread,
            "dropout": dropout,
            "risk": risk,
        }


def score_with_model(
    spec: ClassicSpec,
    model: VectorMemoryModel,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> dict[str, float]:
    state = np.asarray(initial_state, dtype=float).copy()
    states: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for action in actions:
        state, diag = model.predict_next(state, int(action))
        states.append(state.copy())
        diagnostics.append(diag)
    proxy = shaped_utility(spec, states, actions)
    mean_risk = float(np.mean([item["risk"] for item in diagnostics]))
    if model.strategy == "naive_memory":
        proxy += 4.0 * mean_risk
    if model.strategy == "repaired_memory":
        proxy -= 5.0 * mean_risk
    return {
        "proxy_score": float(proxy),
        "selected_risk": mean_risk,
        "retrieval_precision": float(np.mean([item["precision"] for item in diagnostics])),
        "retrieval_stale_rate": float(np.mean([item["stale_rate"] for item in diagnostics])),
        "dropout_variance": float(np.mean([item["dropout"] for item in diagnostics])),
        "retrieval_spread": float(np.mean([item["spread"] for item in diagnostics])),
    }


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
        "selected_risk": float(np.dot(weights, ordered[f"{strategy}_risk"].to_numpy(dtype=float))),
        "selected_precision": float(np.dot(weights, ordered[f"{strategy}_precision"].to_numpy(dtype=float))),
        "selected_stale_rate": float(np.dot(weights, ordered[f"{strategy}_stale"].to_numpy(dtype=float))),
    }


def build_candidate_pool(spec: ClassicSpec, seed: int, *, candidate_count: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    initial_state = start_state(spec, seed)
    memory = build_memory(spec, seed + 1000)
    models = {strategy: VectorMemoryModel(spec, memory, strategy) for strategy in STRATEGIES}
    rows: list[dict[str, float | int | str]] = []
    for candidate_id in range(candidate_count):
        actions = sample_actions(spec, rng)
        true_return, true_diag = execute_true(spec, initial_state, actions)
        row: dict[str, float | int | str] = {
            "benchmark": spec.name,
            "seed": int(seed),
            "candidate_id": int(candidate_id),
            "true_return": float(true_return),
            "terminated_at": float(true_diag["terminated_at"]),
            "final_state_norm": float(true_diag["final_state_norm"]),
        }
        for strategy, model in models.items():
            scored = score_with_model(spec, model, initial_state, actions)
            row[f"{strategy}_proxy"] = scored["proxy_score"]
            row[f"{strategy}_risk"] = scored["selected_risk"]
            row[f"{strategy}_precision"] = scored["retrieval_precision"]
            row[f"{strategy}_stale"] = scored["retrieval_stale_rate"]
            row[f"{strategy}_dropout"] = scored["dropout_variance"]
            row[f"{strategy}_spread"] = scored["retrieval_spread"]
        rows.append(row)
    return pd.DataFrame(rows)


def _ci(values: list[float], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    draws = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975))}


def summarize(curves: pd.DataFrame, effects: pd.DataFrame) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    mismatch_count = 0
    repair_count = 0
    precision_count = 0
    for benchmark in TASKS:
        local = effects[effects["benchmark"] == benchmark]
        proxy_ci = _ci(local["naive_proxy_delta_n64_vs_n1"].tolist(), seed=3100)
        true_ci = _ci(local["naive_true_delta_n64_vs_n1"].tolist(), seed=3200)
        repair_ci = _ci(local["repair_true_gain_n64_over_naive"].tolist(), seed=3300)
        precision_ci = _ci(local["repair_precision_gain_n64_over_naive"].tolist(), seed=3400)
        mismatch = bool(proxy_ci["lo"] > 0.5 and true_ci["hi"] < 0.0)
        repair = bool(repair_ci["lo"] > 0.25)
        precision = bool(precision_ci["lo"] > 0.05)
        mismatch_count += int(mismatch)
        repair_count += int(repair)
        precision_count += int(precision)
        diagnostics[benchmark] = {
            "naive_proxy_delta_n64_vs_n1_ci": proxy_ci,
            "naive_true_delta_n64_vs_n1_ci": true_ci,
            "repair_true_gain_n64_over_naive_ci": repair_ci,
            "repair_precision_gain_n64_over_naive_ci": precision_ci,
            "selected_tail_mismatch": mismatch,
            "repair_helped": repair,
            "repair_improved_precision": precision,
        }
    return {
        "experiment": "v4_gymnasium_memory_benchmarks",
        "benchmarks": list(TASKS),
        "strategies": list(STRATEGIES),
        "n_values": list(N_VALUES),
        "curve_rows": int(len(curves)),
        "effect_rows": int(len(effects)),
        "key_result": {
            "selected_tail_mismatch_benchmark_count": int(mismatch_count),
            "repair_improvement_benchmark_count": int(repair_count),
            "repair_precision_improvement_benchmark_count": int(precision_count),
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
    labels = {
        "recent_only": "recent-only",
        "naive_memory": "naive memory",
        "repaired_memory": "repaired memory",
        "oracle_memory": "oracle memory",
    }
    for axis, benchmark in zip(axes, TASKS):
        sub = means[means["benchmark"] == benchmark]
        for strategy in STRATEGIES:
            local = sub[sub["strategy"] == strategy].sort_values("N")
            axis.plot(local["N"], local["selected_true_return"], marker="o", label=labels[strategy], color=colors[strategy])
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(N_VALUES))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_title(benchmark)
        axis.set_xlabel("candidate budget N")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("expected executed utility")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    path = OUT / "v4_gymnasium_memory_benchmarks.pdf"
    fig.savefig(path)
    shutil.copy2(path, PAPER_FIG / path.name)
    plt.close(fig)
    return path


def write_macros(payload: dict[str, object], curves: pd.DataFrame, effects: pd.DataFrame) -> None:
    key = payload["key_result"]  # type: ignore[index]
    max_n = max(N_VALUES)
    main = curves[curves["N"] == max_n]
    naive = main[main["strategy"] == "naive_memory"]["selected_true_return"].mean()
    repaired = main[main["strategy"] == "repaired_memory"]["selected_true_return"].mean()
    precision_gain = effects["repair_precision_gain_n64_over_naive"].mean()
    text = ""
    text += _macro("GymBenchmarks", len(TASKS))
    text += _macro("GymCurveRows", len(curves))
    text += _macro("GymEffectRows", len(effects))
    text += _macro("GymMismatchCount", key["selected_tail_mismatch_benchmark_count"])  # type: ignore[index]
    text += _macro("GymRepairCount", key["repair_improvement_benchmark_count"])  # type: ignore[index]
    text += _macro("GymPrecisionCount", key["repair_precision_improvement_benchmark_count"])  # type: ignore[index]
    text += _macro("GymNaiveMaxTrueMean", _fmt(naive, 2))
    text += _macro("GymRepairedMaxTrueMean", _fmt(repaired, 2))
    text += _macro("GymMeanRepairGain", _fmt(repaired - naive, 2))
    text += _macro("GymMeanPrecisionGain", _pct(precision_gain))
    (PAPER / "v4_gym_macros.tex").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_frames: list[pd.DataFrame] = []
    curve_rows: list[dict[str, float | int | str]] = []
    effect_rows: list[dict[str, float | int | str]] = []
    seeds = [17, 29]
    for benchmark, spec in TASKS.items():
        for seed in seeds:
            candidates = build_candidate_pool(spec, seed)
            candidate_frames.append(candidates)
            for strategy in STRATEGIES:
                for n_value in N_VALUES:
                    row = selection_expectation(candidates, strategy, n_value)
                    row.update({"benchmark": benchmark, "seed": int(seed)})
                    curve_rows.append(row)
            local_curves = pd.DataFrame([row for row in curve_rows if row["benchmark"] == benchmark and row["seed"] == seed])
            n1 = local_curves[local_curves["N"] == 1].set_index("strategy")
            n64 = local_curves[local_curves["N"] == 64].set_index("strategy")
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
                    "repair_true_gain_n64_over_naive": float(
                        n64.loc["repaired_memory", "selected_true_return"] - n64.loc["naive_memory", "selected_true_return"]
                    ),
                    "repair_precision_gain_n64_over_naive": float(
                        n64.loc["repaired_memory", "selected_precision"] - n64.loc["naive_memory", "selected_precision"]
                    ),
                    "oracle_gap_n64_over_repair": float(
                        n64.loc["oracle_memory", "selected_true_return"] - n64.loc["repaired_memory", "selected_true_return"]
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
        "candidates": "results/v4_gymnasium_memory/candidate_records.csv",
        "curves": "results/v4_gymnasium_memory/selection_curves.csv",
        "effects": "results/v4_gymnasium_memory/benchmark_effects.csv",
        "figure": "paper/figures/v4_gymnasium_memory_benchmarks.pdf",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_macros(payload, curves, effects)
    print(f"v4 Gymnasium memory benchmarks complete: {OUT}")
    print(
        "benchmarks={benchmarks} mismatch={mismatch} repair={repair} precision={precision} figure={figure}".format(
            benchmarks=len(TASKS),
            mismatch=payload["key_result"]["selected_tail_mismatch_benchmark_count"],  # type: ignore[index]
            repair=payload["key_result"]["repair_improvement_benchmark_count"],  # type: ignore[index]
            precision=payload["key_result"]["repair_precision_improvement_benchmark_count"],  # type: ignore[index]
            figure=figure,
        )
    )


if __name__ == "__main__":
    main()
