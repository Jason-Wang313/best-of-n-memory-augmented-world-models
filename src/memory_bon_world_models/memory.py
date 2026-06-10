from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import FORWARD, REVERSE, Regime, RegimeWorld


@dataclass(frozen=True)
class Transition:
    state: float
    action: float
    next_state: float
    delta: float
    regime: str
    timestamp: float
    source: str

    @property
    def is_recent(self) -> bool:
        return self.timestamp >= 1000.0


@dataclass(frozen=True)
class RetrievalResult:
    transitions: tuple[Transition, ...]
    distances: np.ndarray
    weights: np.ndarray
    policy: str

    def mean_delta(self) -> float:
        if not self.transitions:
            return 0.0
        deltas = np.asarray([transition.delta for transition in self.transitions], dtype=float)
        return float(np.average(deltas, weights=self.weights))

    def stale_rate(self) -> float:
        if not self.transitions:
            return 1.0
        return float(np.mean([not transition.is_recent for transition in self.transitions]))

    def precision(self, query_regime: str | None) -> float:
        if query_regime is None or not self.transitions:
            return np.nan
        return float(np.mean([transition.regime == query_regime for transition in self.transitions]))

    def sign_disagreement(self) -> float:
        usable = [
            np.sign(transition.delta / transition.action)
            for transition in self.transitions
            if abs(transition.action) > 1e-8
        ]
        if len(usable) <= 1:
            return 0.0
        signs = np.asarray(usable, dtype=float)
        return float(1.0 - abs(np.mean(signs)))


class EpisodicMemory:
    """A transition memory with intentionally observable age metadata."""

    def __init__(self, transitions: list[Transition]):
        self.transitions = list(transitions)
        self.states = np.asarray([t.state for t in self.transitions], dtype=float)
        self.actions = np.asarray([t.action for t in self.transitions], dtype=float)
        self.deltas = np.asarray([t.delta for t in self.transitions], dtype=float)
        self.regimes = np.asarray([t.regime for t in self.transitions], dtype=object)
        self.is_recent = np.asarray([t.is_recent for t in self.transitions], dtype=bool)

    @staticmethod
    def _feature(state: float, action: float) -> np.ndarray:
        return np.asarray([float(state), 0.85 * float(action)], dtype=float)

    @classmethod
    def from_regime_mix(
        cls,
        rng: np.random.Generator,
        *,
        current_regime: Regime,
        stale_regime: Regime = FORWARD,
        stale_fraction: float = 0.85,
        total: int = 900,
        state_scale: float = 1.35,
        world: RegimeWorld | None = None,
    ) -> "EpisodicMemory":
        """Create dense old memory plus sparse recent calibration memory."""

        if world is None:
            world = RegimeWorld()
        stale_n = int(round(total * stale_fraction))
        recent_n = max(16, total - stale_n)
        transitions: list[Transition] = []

        def add_block(n: int, regime: Regime, source: str, time_low: float, time_high: float) -> None:
            for _ in range(n):
                state = float(rng.normal(0.0, state_scale))
                action = float(rng.choice(world.action_space, p=np.array([0.42, 0.16, 0.42])))
                next_state = world.step(state, action, regime)
                # Keep memory imperfect but not pathological.
                observed_next = float(next_state + rng.normal(0.0, 0.025))
                transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        next_state=observed_next,
                        delta=observed_next - state,
                        regime=regime.name,
                        timestamp=float(rng.uniform(time_low, time_high)),
                        source=source,
                    )
                )

        add_block(stale_n, stale_regime, "stale", 0.0, 100.0)
        add_block(recent_n, current_regime, "recent", 1000.0, 1100.0)
        rng.shuffle(transitions)
        return cls(transitions)

    def retrieve(
        self,
        state: float,
        action: float,
        *,
        k: int = 9,
        policy: str = "naive",
        query_regime: str | None = None,
    ) -> RetrievalResult:
        state_diff = self.states - float(state)
        action_diff = 0.85 * (self.actions - float(action))
        squared_distances = state_diff * state_diff + action_diff * action_diff
        scores = squared_distances.copy()

        if policy == "recency":
            age_penalty = np.where(self.is_recent, 0.0, 1.25)
            scores = squared_distances + age_penalty
        elif policy == "oracle":
            if query_regime is None:
                raise ValueError("oracle retrieval requires query_regime")
            scores = squared_distances + np.where(self.regimes == query_regime, 0.0, 1e6)
        elif policy != "naive":
            raise ValueError(f"unknown retrieval policy: {policy}")

        k = min(int(k), len(self.transitions))
        unordered = np.argpartition(scores, k - 1)[:k]
        order = unordered[np.argsort(scores[unordered])]
        picked = tuple(self.transitions[int(i)] for i in order)
        picked_distances = np.sqrt(squared_distances[order])
        weights = 1.0 / (picked_distances + 1e-3)
        weights = weights / np.sum(weights)
        return RetrievalResult(picked, picked_distances, weights, policy)

    def recent_slope(self) -> float:
        """Fit a local non-memory action-effect estimate from recent calibration."""

        xs = []
        ys = []
        for transition in self.transitions:
            if transition.is_recent and abs(transition.action) > 1e-8:
                xs.append(transition.action)
                ys.append(transition.delta)
        if not xs:
            return 0.0
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(x, y) / denom)


def canonical_reverse_memory(seed: int = 0, stale_fraction: float = 0.88) -> EpisodicMemory:
    rng = np.random.default_rng(seed)
    return EpisodicMemory.from_regime_mix(rng, current_regime=REVERSE, stale_fraction=stale_fraction)
