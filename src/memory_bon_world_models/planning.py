from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Regime, RegimeWorld
from .models import BaseWorldModel, RepairedMemoryWorldModel


@dataclass(frozen=True)
class CandidateScore:
    actions: np.ndarray
    proxy_score: float
    predicted_final: float
    predicted_final_error: float
    mean_stale_rate: float
    mean_precision: float
    mean_gate: float
    mean_disagreement: float
    mean_dropout_variance: float
    mean_risk: float


def sample_action_sequences(rng: np.random.Generator, n: int, horizon: int) -> np.ndarray:
    """Sample candidate controls with both directional modes represented."""

    sequences = np.zeros((int(n), int(horizon)), dtype=float)
    for i in range(int(n)):
        mode = float(rng.choice([-1.0, 1.0]))
        probabilities = {
            -mode: 0.14,
            0.0: 0.16,
            mode: 0.70,
        }
        actions = np.asarray(list(probabilities.keys()), dtype=float)
        probs = np.asarray(list(probabilities.values()), dtype=float)
        sequences[i] = rng.choice(actions, size=int(horizon), p=probs)
    return sequences


def score_candidate(
    model: BaseWorldModel,
    world: RegimeWorld,
    regime: Regime,
    actions: np.ndarray,
    *,
    repair_penalty: float = 0.0,
) -> CandidateScore:
    predicted_state = 0.0
    diagnostics = []
    for action in actions:
        prediction = model.predict_next(predicted_state, float(action), query_regime=regime)
        diagnostics.append(prediction)
        predicted_state = prediction.next_state

    proxy = world.reward_from_final(predicted_state, actions)
    risks = np.asarray([diag.risk for diag in diagnostics], dtype=float)
    if repair_penalty > 0.0:
        proxy -= repair_penalty * float(np.mean(risks))

    precision_values = np.asarray([diag.retrieval_precision for diag in diagnostics], dtype=float)
    mean_precision = float(np.nanmean(precision_values)) if np.any(~np.isnan(precision_values)) else np.nan

    return CandidateScore(
        actions=np.asarray(actions, dtype=float),
        proxy_score=float(proxy),
        predicted_final=float(predicted_state),
        predicted_final_error=abs(world.goal - float(predicted_state)),
        mean_stale_rate=float(np.mean([diag.stale_rate for diag in diagnostics])),
        mean_precision=mean_precision,
        mean_gate=float(np.mean([diag.gate for diag in diagnostics])),
        mean_disagreement=float(np.mean([diag.disagreement for diag in diagnostics])),
        mean_dropout_variance=float(np.mean([diag.dropout_variance for diag in diagnostics])),
        mean_risk=float(np.mean(risks)),
    )


def plan_best_of_n(
    rng: np.random.Generator,
    model: BaseWorldModel,
    world: RegimeWorld,
    regime: Regime,
    *,
    n: int,
    repair_penalty: float = 0.0,
) -> dict[str, float | str]:
    candidates = sample_action_sequences(rng, int(n), world.horizon)
    penalty = repair_penalty if isinstance(model, RepairedMemoryWorldModel) else 0.0
    scored = [score_candidate(model, world, regime, actions, repair_penalty=penalty) for actions in candidates]
    selected = max(scored, key=lambda item: item.proxy_score)
    true = world.evaluate(selected.actions, regime)
    proxy_true_gap = float(selected.proxy_score - true["true_return"])
    hallucinated = float((selected.proxy_score > -1.75) and (true["true_return"] < -5.0))

    return {
        "strategy": model.name,
        "n": int(n),
        "proxy_score": selected.proxy_score,
        "predicted_final": selected.predicted_final,
        "predicted_final_error": selected.predicted_final_error,
        "true_return": float(true["true_return"]),
        "true_final": float(true["true_final"]),
        "true_final_error": float(true["true_final_error"]),
        "true_success": float(true["true_success"]),
        "proxy_true_gap": proxy_true_gap,
        "hallucinated_rollout": hallucinated,
        "mean_selected_action": float(np.mean(selected.actions)),
        "retrieval_stale_rate": selected.mean_stale_rate,
        "retrieval_precision": selected.mean_precision,
        "memory_gate": selected.mean_gate,
        "retrieval_disagreement": selected.mean_disagreement,
        "dropout_variance": selected.mean_dropout_variance,
        "diagnostic_risk": selected.mean_risk,
    }
