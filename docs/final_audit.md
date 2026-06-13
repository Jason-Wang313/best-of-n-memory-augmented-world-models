# Final Audit

## Main Thesis

Max-score N-candidate selection can amplify stale or support-mismatched
retrieval errors in memory-augmented world models. When stale memories make an
incorrect rollout look high-value under the model proxy, increasing `N`
increases the chance that the selector chooses that memory-impostor rollout.

## Genuine Novelty

The project is not novel as a memory-augmented world-model architecture. The
novelty claim is narrower and more defensible: an architecture-specific
mechanism, diagnostic, and repair for the interaction between retrieval memory
and N-candidate max-score rollout selection.

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

In the completed v2 paper run at stale fraction 0.85 and `N=64`, naive
memory-augmented max-score selected rollouts reached mean true return `-13.334`
while its model proxy score was `-2.591`, producing a proxy-true gap of `10.743`.
The non-memory baseline reached `-0.180`, and the oracle memory retriever reached
`-1.728`.

## Strongest Diagnostic Result

At `N=64`, naive memory selection had retrieval precision `15.7%`, stale
retrieval rate `84.3%`, and hallucinated-rollout rate `5.6%`. These diagnostics
match the proposed failure mechanism: the selected candidate is not merely a bad
random action sequence; it is supported by stale retrieved transitions.

## Strongest Repair Result

The repaired memory model uses recency-aware retrieval, disagreement-aware
memory gating, and memory-dropout risk penalties without access to the hidden
regime label. At `N=64`, it improved true return from naive memory's `-13.334`
to `-1.897` and reduced hallucinated-rollout rate to `0.0%`.

## Biggest Weaknesses

- The benchmark is synthetic and one-dimensional.
- The v2 submission artifact should use the bounded multi-staleness
  `results/v2paper/` preset when present. The larger `results/paper/` preset
  remains a slower optional scale-up.
- The repair is intentionally simple and should be treated as a diagnostic
  intervention rather than a production-grade retrieval policy.
- The paper does not validate on neural simulators, robotics environments, or
  large memory stores.

## Paper-Readiness Judgment

Ready as a scoped mechanism paper only if the manuscript foregrounds the
memory-impostor mechanism, keeps external-validity limitations visible, and
uses the strongest completed experiment preset available in the repository.

## Verification

- Unit tests: `pytest -q` passed with 8 tests.
- Literature generation: `python scripts/build_literature.py` passed.
- V2 paper experiment: `python experiments/run_synthetic.py --preset v2paper`
  passed and generated the paper-facing CSV results plus figures.
- Paper build: `python scripts/build_paper.py` passed using `pdflatex`/BibTeX
  fallback after `latexmk` was unavailable due missing Perl.
- Anonymous ICLR style: `paper/main.tex` uses `iclr2026_conference,times`, keeps
  `\iclrfinalcopy` commented, and declares `Anonymous Authors`.

## Exact PDF Path

`C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models-v2.pdf`

## GitHub Repo URL

https://github.com/Jason-Wang313/best-of-n-memory-augmented-world-models
