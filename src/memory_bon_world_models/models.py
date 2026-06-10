from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Regime
from .memory import EpisodicMemory, RetrievalResult


@dataclass(frozen=True)
class Prediction:
    next_state: float
    memory_delta: float
    base_delta: float
    gate: float
    stale_rate: float
    retrieval_precision: float
    disagreement: float
    dropout_variance: float
    risk: float


class BaseWorldModel:
    """Non-memory dynamics model fit from recent calibration transitions."""

    name = "base_bon"
    uses_latent_regime = False

    def __init__(self, memory: EpisodicMemory, *, k: int = 9):
        self.memory = memory
        self.k = int(k)
        self.effect = float(memory.recent_slope())

    def predict_next(self, state: float, action: float, *, query_regime: Regime | None = None) -> Prediction:
        base_delta = self.effect * float(action)
        next_state = float(state + base_delta)
        return Prediction(
            next_state=next_state,
            memory_delta=0.0,
            base_delta=base_delta,
            gate=0.0,
            stale_rate=0.0,
            retrieval_precision=np.nan,
            disagreement=0.0,
            dropout_variance=0.0,
            risk=0.0,
        )


class MemoryWorldModel(BaseWorldModel):
    """Naive retrieval-augmented transition model."""

    name = "memory_bon"
    uses_latent_regime = False

    def __init__(self, memory: EpisodicMemory, *, k: int = 9, gate: float = 0.86):
        super().__init__(memory, k=k)
        self.memory_gate = float(gate)

    def _retrieve(self, state: float, action: float, query_regime: Regime | None) -> RetrievalResult:
        regime_name = None if query_regime is None else query_regime.name
        return self.memory.retrieve(state, action, k=self.k, policy="naive", query_regime=regime_name)

    def _dropout_variance(self, retrieval: RetrievalResult) -> float:
        deltas = np.asarray([transition.delta for transition in retrieval.transitions], dtype=float)
        if len(deltas) <= 2:
            return 0.0
        leave_one_out = []
        weights = np.asarray(retrieval.weights, dtype=float)
        for i in range(len(deltas)):
            mask = np.ones(len(deltas), dtype=bool)
            mask[i] = False
            local_w = weights[mask]
            local_w = local_w / np.sum(local_w)
            leave_one_out.append(float(np.average(deltas[mask], weights=local_w)))
        return float(np.var(leave_one_out))

    def predict_next(self, state: float, action: float, *, query_regime: Regime | None = None) -> Prediction:
        retrieval = self._retrieve(state, action, query_regime)
        memory_delta = retrieval.mean_delta()
        base_delta = self.effect * float(action)
        gate = self.memory_gate
        blended_delta = (1.0 - gate) * base_delta + gate * memory_delta
        regime_name = None if query_regime is None else query_regime.name
        stale = retrieval.stale_rate()
        disagreement = retrieval.sign_disagreement()
        dropout_var = self._dropout_variance(retrieval)
        risk = stale + disagreement + min(2.0, 12.0 * dropout_var)
        return Prediction(
            next_state=float(state + blended_delta),
            memory_delta=memory_delta,
            base_delta=base_delta,
            gate=gate,
            stale_rate=stale,
            retrieval_precision=retrieval.precision(regime_name),
            disagreement=disagreement,
            dropout_variance=dropout_var,
            risk=float(risk),
        )


class OracleMemoryWorldModel(MemoryWorldModel):
    """Oracle retriever for upper-bound comparison.

    This model is intentionally marked as using the hidden latent regime and is
    never used as the proposed repair.
    """

    name = "oracle_memory_bon"
    uses_latent_regime = True

    def _retrieve(self, state: float, action: float, query_regime: Regime | None) -> RetrievalResult:
        if query_regime is None:
            raise ValueError("oracle model requires the true query regime")
        return self.memory.retrieve(state, action, k=self.k, policy="oracle", query_regime=query_regime.name)


class RepairedMemoryWorldModel(MemoryWorldModel):
    """Memory model with age-aware retrieval and uncertainty-aware gating."""

    name = "repaired_memory_bon"
    uses_latent_regime = False

    def __init__(self, memory: EpisodicMemory, *, k: int = 11, max_gate: float = 0.78):
        super().__init__(memory, k=k, gate=max_gate)
        self.max_gate = float(max_gate)

    def _retrieve(self, state: float, action: float, query_regime: Regime | None) -> RetrievalResult:
        regime_name = None if query_regime is None else query_regime.name
        return self.memory.retrieve(state, action, k=self.k, policy="recency", query_regime=regime_name)

    def predict_next(self, state: float, action: float, *, query_regime: Regime | None = None) -> Prediction:
        retrieval = self._retrieve(state, action, query_regime)
        memory_delta = retrieval.mean_delta()
        base_delta = self.effect * float(action)
        stale = retrieval.stale_rate()
        disagreement = retrieval.sign_disagreement()
        dropout_var = self._dropout_variance(retrieval)

        uncertainty = stale + 0.85 * disagreement + min(2.0, 14.0 * dropout_var)
        gate = float(np.clip(self.max_gate * (1.0 - 0.55 * uncertainty), 0.05, self.max_gate))
        blended_delta = (1.0 - gate) * base_delta + gate * memory_delta
        regime_name = None if query_regime is None else query_regime.name
        return Prediction(
            next_state=float(state + blended_delta),
            memory_delta=memory_delta,
            base_delta=base_delta,
            gate=gate,
            stale_rate=stale,
            retrieval_precision=retrieval.precision(regime_name),
            disagreement=disagreement,
            dropout_variance=dropout_var,
            risk=float(uncertainty),
        )
