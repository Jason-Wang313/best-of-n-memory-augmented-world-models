# Final Audit

## Main Thesis

Best-of-N inference can amplify stale or support-mismatched retrieval errors in
memory-augmented world models. When stale memories make an incorrect rollout
look high-value under the model proxy, increasing `N` increases the chance that
the selector chooses that memory-impostor rollout.

## Genuine Novelty

The project is not novel as a memory-augmented world-model architecture. The
novelty claim is narrower and more defensible: an architecture-specific
mechanism, diagnostic, and repair for the interaction between retrieval memory
and Best-of-N rollout selection.

## Literature Coverage

- `docs/related_work_matrix.csv` contains 129 entries.
- Required columns are present: title, year, venue/arXiv, link, cluster,
  contribution, relevance score, threat level, and why it does or does not
  subsume this project.
- The hostile-prior memo identifies 10 reviewer threat papers.
- The novelty decision rejects an architecture claim and commits to a mechanism
  and diagnostic claim.

## Proof Status

The only formal claim is a finite-N two-type proposition: if memory-impostor
candidates have higher proxy score and lower true utility than grounded
candidates, the probability of selecting an impostor is `1-(1-p)^N`, and
expected true utility decreases monotonically with `N`. This is a mechanism
statement, not a theorem about arbitrary neural retrievers.

## Strongest Empirical Result

In the completed smoke run at stale fraction 0.85 and `N=64`, naive
memory-augmented Best-of-N selected rollouts with mean true return `-13.287`
while its model proxy score was `-2.403`, producing a proxy-true gap of `10.884`.
The non-memory baseline reached `-0.187`, and the oracle memory retriever reached
`-1.553`.

## Strongest Diagnostic Result

At `N=64`, naive memory selection had retrieval precision `13.7%`, stale
retrieval rate `86.3%`, and hallucinated-rollout rate `27.8%`. These diagnostics
match the proposed failure mechanism: the selected candidate is not merely a bad
random action sequence; it is supported by stale retrieved transitions.

## Strongest Repair Result

The repaired memory model uses recency-aware retrieval, disagreement-aware
memory gating, and memory-dropout risk penalties without access to the hidden
regime label. At `N=64`, it improved true return from naive memory's `-13.287`
to `-1.954` and reduced hallucinated-rollout rate to `0.0%`.

## Biggest Weaknesses

- The benchmark is synthetic and one-dimensional.
- The completed empirical run is the smoke preset; the larger multi-staleness
  paper preset was not run because the smoke preset took about two minutes on
  this machine and the full preset is estimated to be tens of minutes.
- The repair is intentionally simple and should be treated as a diagnostic
  intervention rather than a production-grade retrieval policy.
- The paper does not validate on neural simulators, robotics environments, or
  large memory stores.

## Paper-Readiness Judgment

Ready as a first autonomous ICLR-style research package, with honest claim
boundaries. It is not yet ready as a competitive final submission without a
larger empirical section, stronger baselines, and at least one realistic
memory-augmented world-model setting.

## Verification

- Unit tests: `pytest -q` passed with 8 tests.
- Literature generation: `python scripts/build_literature.py` passed.
- Smoke experiment: `python experiments/run_synthetic.py --preset smoke` passed
  and generated CSV results plus figures.
- Paper build: `python scripts/build_paper.py` passed using `pdflatex`/BibTeX
  fallback after `latexmk` was unavailable due missing Perl.
- Anonymous ICLR style: `paper/main.tex` uses `iclr2026_conference,times`, keeps
  `\iclrfinalcopy` commented, and declares `Anonymous Authors`.

## Exact PDF Path

`C:\Users\wangz\Downloads\best-of-n-memory-augmented-world-models.pdf`

## GitHub Repo URL

https://github.com/Jason-Wang313/best-of-n-memory-augmented-world-models
