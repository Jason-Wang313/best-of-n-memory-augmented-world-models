# best-of-n-memory-augmented-world-models v3 Review Audit

## Scope

This audit records the v3 self-review for the memory-augmented world-model
paper. The manuscript is the memory-impostor rollout paper: stale or
support-mismatched retrieval creates high-proxy, low-true rollouts, and
candidate-budget max-score selection amplifies those specific candidates.

Source of truth:

- Local folder: `C:\Users\wangz\best-of-n-memory-augmented-world-models`
- GitHub repo: `Jason-Wang313/best-of-n-memory-augmented-world-models`
- Desktop artifact: `C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models-v3.pdf`
- Repo artifact: `paper/final/best-of-n-memory-augmented-world-models-v3.pdf`

## V3 Additions

- Added `results/v3_base/` as the current paper-facing checked-in CSV source.
- Added `experiments/18_v3_cached_evidence.py` for bootstrap intervals,
  budget-delta ledgers, trial-level harm rates, risk correlations, and v3
  figures/macros.
- Expanded the manuscript to 25 pages with stress evidence, evidence tiers,
  selected-tail decomposition, negative controls, repair/oracle boundaries,
  transfer protocol, reviewer attack ledger, and archive contract.
- Updated the build to regenerate cached evidence and export a v3 PDF to both
  Desktop and `paper/final/`.
- Added `scripts/run_v3_claim_audit.py`.

## Harsh Reviewer Stance

The paper should be rejected if it claims broad deployed-agent safety,
production-grade repair, robotics transfer, visual-control validation, or a
general theorem about arbitrary neural retrievers. The v3 paper does not make
those claims.

The paper is defensible as a scoped mechanism submission because it provides a
clean failure generator, selected-rollout provenance diagnostics, non-memory
and oracle controls, a non-oracle repair, bootstrap intervals, paired harm
rates, and a reproducible v3 artifact package.

Residual risk is external validity, not duplicate-wrapper identity.
