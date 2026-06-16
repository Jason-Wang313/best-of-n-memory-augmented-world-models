# best-of-n-memory-augmented-world-models v4 Review Audit

## Scope

This audit records the v4 self-review for the memory-augmented world-model
paper. The manuscript is the memory-impostor rollout paper: stale or
support-mismatched retrieval creates high-proxy, low-true rollouts, and
candidate-budget max-score selection amplifies those specific candidates.

Source of truth:

- Local folder: `C:\Users\wangz\best-of-n-memory-augmented-world-models`
- GitHub repo: `Jason-Wang313/best-of-n-memory-augmented-world-models`
- Desktop artifact: `C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models-v4.pdf`
- Repo artifact: `paper/final/best-of-n-memory-augmented-world-models-v4.pdf`

## V4 Additions

- Added `results/v4_base/` as the current paper-facing checked-in CSV source.
- Added `experiments/18_v4_protocol_evidence.py` for bootstrap intervals,
  budget-delta ledgers, trial-level harm rates, risk correlations, and v4
  figures/macros.
- Added exact Gymnasium toy-text benchmark cards in
  `experiments/20_v4_toytext_memory_benchmarks.py`: FrozenLake-v1,
  CliffWalkingSlippery-v1, and Taxi-v3.
- Added executable Gymnasium classic-control boundary cards in
  `experiments/19_v4_gymnasium_memory_benchmarks.py`: CartPole-v1,
  MountainCar-v0, and Acrobot-v1.
- Expanded the manuscript to 25 pages with stress evidence, evidence tiers,
  selected-tail decomposition, negative controls, repair/oracle boundaries,
  transfer protocol, reviewer attack ledger, and archive contract.
- Updated the build to regenerate protocol evidence and export a v4 PDF to both
  Desktop and `paper/final/`.
- Added `scripts/run_v4_claim_audit.py`.

## Harsh Reviewer Stance

The paper should be rejected if it claims broad deployed-agent safety,
production-grade repair, robotics transfer, visual-control validation, or a
general theorem about arbitrary neural retrievers. The v4 paper does not make
those claims.

The paper is defensible as a scoped mechanism submission because it provides a
clean failure generator, selected-rollout provenance diagnostics, non-memory
and oracle controls, a non-oracle repair, bootstrap intervals, paired harm
rates, exact standard toy-text transfer cards, classic-control boundary cards,
and a reproducible v4 artifact package.

Residual risk is broad external validity, not duplicate-wrapper identity. The
paper still must not claim D4RL, Atari, visual-control, robotics, or SOTA RL
coverage from the lightweight Gymnasium cards.
