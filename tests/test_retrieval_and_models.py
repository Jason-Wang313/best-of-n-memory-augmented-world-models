import numpy as np

from memory_bon_world_models.environment import REVERSE
from memory_bon_world_models.memory import canonical_reverse_memory
from memory_bon_world_models.models import OracleMemoryWorldModel, RepairedMemoryWorldModel


def test_naive_retrieval_is_stale_under_constructed_shift():
    memory = canonical_reverse_memory(seed=3, stale_fraction=0.9)
    result = memory.retrieve(0.0, 1.0, k=15, policy="naive", query_regime=REVERSE.name)
    assert result.stale_rate() > 0.65
    assert result.precision(REVERSE.name) < 0.35


def test_oracle_retrieval_uses_latent_only_for_upper_bound():
    memory = canonical_reverse_memory(seed=5, stale_fraction=0.9)
    oracle = OracleMemoryWorldModel(memory)
    repaired = RepairedMemoryWorldModel(memory)
    assert oracle.uses_latent_regime is True
    assert repaired.uses_latent_regime is False

    pred = oracle.predict_next(0.0, 1.0, query_regime=REVERSE)
    assert pred.retrieval_precision > 0.99
    assert pred.next_state < 0.0


def test_repaired_model_downweights_old_disagreement_without_hidden_label():
    memory = canonical_reverse_memory(seed=11, stale_fraction=0.9)
    repaired = RepairedMemoryWorldModel(memory)
    pred = repaired.predict_next(0.0, 1.0, query_regime=REVERSE)
    assert pred.gate <= 0.78
    assert pred.retrieval_precision > 0.7
    assert np.isfinite(pred.dropout_variance)
