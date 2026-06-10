# Novelty Decision

## Decision

Proceed as a standalone mechanism/diagnostic paper, not as a new architecture paper.

## Reasoning

The literature sweep found strong prior art on all ingredients in isolation:
model-based RL and world models, episodic memory and case-based control,
retrieval-augmented neural models, uncertainty-aware model-based planning, and
Best-of-N/test-time selection. A claim that memory-augmented world models or
retrieval-based planning are new would be indefensible.

The remaining gap is the interaction effect: when a retrieval-augmented dynamics
model has stale or support-mismatched cases, Best-of-N inference can select rare
candidate rollouts that look excellent under the memory-biased model and are bad
under the true dynamics. The project therefore centers on:

1. a finite-N mechanism statement for memory-impostor amplification;
2. a diagnostic suite measuring retrieval precision, staleness, dropout
   sensitivity, proxy-true gap, and hallucinated rollout rate;
3. a controlled synthetic benchmark where true dynamics and memory provenance
   are known;
4. a repair based on recency-aware retrieval, disagreement-aware gating, and
   uncertainty penalties.

## Claim Boundary

The paper should not claim that all memory-augmented world models fail under
Best-of-N. It should claim that this failure mode is real in a controlled setting,
mechanistically predictable, measurable, and mitigated by a principled repair.
