from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regime:
    """A one-dimensional latent dynamics regime."""

    name: str
    action_effect: float
    drift: float = 0.0
    process_noise: float = 0.0


FORWARD = Regime("forward", action_effect=1.0)
REVERSE = Regime("reverse", action_effect=-1.0)
SLOW_FORWARD = Regime("slow_forward", action_effect=0.45)


class RegimeWorld:
    """Small latent-regime world with a transparent true transition function.

    The observation is only the scalar position. The latent regime determines
    the sign and magnitude of action effects, but the naive retriever does not
    observe that regime. This makes it possible for dense stale memories from a
    previous regime to dominate nearest-neighbor transition prediction.
    """

    action_space = np.array([-1.0, 0.0, 1.0])

    def __init__(self, goal: float = 6.0, horizon: int = 8, action_cost: float = 0.03):
        self.goal = float(goal)
        self.horizon = int(horizon)
        self.action_cost = float(action_cost)

    def step(self, state: float, action: float, regime: Regime, rng: np.random.Generator | None = None) -> float:
        noise = 0.0
        if regime.process_noise > 0.0 and rng is not None:
            noise = float(rng.normal(0.0, regime.process_noise))
        return float(state + regime.action_effect * action + regime.drift + noise)

    def rollout(
        self,
        actions: np.ndarray,
        regime: Regime,
        *,
        start_state: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        states = [float(start_state)]
        state = float(start_state)
        for action in np.asarray(actions, dtype=float):
            state = self.step(state, float(action), regime, rng)
            states.append(state)
        return np.asarray(states, dtype=float)

    def reward_from_final(self, final_state: float, actions: np.ndarray) -> float:
        distance = abs(self.goal - float(final_state))
        effort = self.action_cost * float(np.sum(np.abs(actions)))
        return -distance - effort

    def evaluate(self, actions: np.ndarray, regime: Regime, *, start_state: float = 0.0) -> dict[str, float]:
        states = self.rollout(actions, regime, start_state=start_state)
        reward = self.reward_from_final(float(states[-1]), actions)
        return {
            "true_return": reward,
            "true_final": float(states[-1]),
            "true_final_error": abs(self.goal - float(states[-1])),
            "true_success": float(abs(self.goal - float(states[-1])) <= 1.5),
        }


def regime_by_name(name: str) -> Regime:
    regimes = {regime.name: regime for regime in (FORWARD, REVERSE, SLOW_FORWARD)}
    return regimes[name]
