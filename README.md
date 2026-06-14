# Memory-Impostor Rollouts

This repository is a v3 anonymous ICLR-style research package on a
narrow failure mode of retrieval-, episodic-memory-, and case-based world
models under max-score rollout selection.

The central question is not whether memory helps world models. It often does.
The question here is what happens when a memory-augmented dynamics model is
queried under support mismatch: the nearest retrieved cases are stale or from a
latent regime with the wrong transition sign, and an N-candidate planner selects
the rollout with the highest model-predicted return.

The package includes:

- a controlled synthetic regime-switching world with measurable latent regimes,
  memory staleness, retrieval precision, and true rollout utility;
- naive, oracle, non-memory, and repaired memory-augmented world models;
- candidate-budget rollout-selection experiments across `N`, staleness, and repair settings;
- publication figures, raw CSV outputs, and v3 cached stress evidence;
- literature sweep and hostile prior-work docs;
- an anonymous ICLR 2026 LaTeX paper source and build script.

## Quick Start

```powershell
cd C:\Users\wangz\best-of-n-memory-augmented-world-models
python -m pip install -e .
pytest
python experiments\18_v3_cached_evidence.py
python scripts\build_literature.py
python scripts\build_paper.py
python scripts\run_v3_claim_audit.py
```

The versioned Desktop PDF target is:

```text
C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models-v3.pdf
```

## Paper Angle

The final contribution is deliberately scoped as a mechanism paper:

1. Max-score N-candidate selection amplifies a rare candidate type whenever
   that type has higher proxy score and lower true value than grounded
   candidates.
2. In memory-augmented world models, stale or support-mismatched retrieval can
   create exactly that candidate type: memory-impostor rollouts.
3. A diagnostic based on retrieval precision, staleness, proxy-true gap, and
   counterfactual memory dropout exposes the mechanism.
4. A simple repair using recency-aware retrieval, disagreement-aware memory
   gating, and dropout-variance penalties mitigates the synthetic failure.

This is not presented as evidence that all deployed memory-augmented agents fail
this way. It is a compact runnable diagnostic for an architectural interaction
that related work usually treats separately.

## Repository Layout

```text
src/memory_bon_world_models/  core environment, memory, models, planning
experiments/                  runnable experiment entry points
scripts/                      literature and paper build scripts
tests/                        reproducibility and mechanism tests
docs/                         literature map, novelty decision, audit
results/                      generated CSV metrics
figures/                      generated plots
paper/                        anonymous ICLR source and style files
```
