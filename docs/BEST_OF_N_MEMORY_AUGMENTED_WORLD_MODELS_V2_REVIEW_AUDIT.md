# best-of-n-memory-augmented-world-models v2 Review Audit

## Scope

This audit records the v2 self-review for the memory-augmented world-model
paper. The main duplicate risk is that neighboring Desktop papers use a generic
"Best-of-N exact law" wrapper. This paper must instead read as the
memory-impostor rollout paper: stale retrieval provenance creates high-proxy,
low-true candidates, and N-candidate max-score selection amplifies those
specific candidates.

Source of truth:

- Local folder: `C:\Users\wangz\best-of-n-memory-augmented-world-models`
- GitHub repo: `Jason-Wang313/best-of-n-memory-augmented-world-models`
- Desktop source PDF: `C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models.pdf`
- Desktop v2 artifact after build: `C:\Users\wangz\OneDrive\Desktop\best-of-n-memory-augmented-world-models-v2.pdf`

## Fixes Applied Before Final Review

- Replaced the title with `Memory-Impostor Rollouts in Retrieval-Augmented
  World Models`.
- Rewrote the abstract around memory-impostor rollouts, provenance
  diagnostics, hallucinated-rollout rate, and non-oracle repair.
- Reframed the introduction and contributions so N-candidate selection is the
  amplification mechanism, not the paper brand.
- Updated README, project description, final audit, and build output path to
  match the v2 identity.
- Kept the formal statement deliberately simple and scoped: finite-$N$
  memory-impostor amplification, not a theorem about arbitrary neural
  retrievers.

## 50-Round Attack Pass

1. Title looks like another Best-of-N wrapper. Fixed: title names
   memory-impostor rollouts and stale retrieval.
2. Abstract opens with generic Best-of-N. Fixed: abstract opens with memory
   retrieval failure under rollout selection.
3. Paper could be confused with LLM score-rank paper. Mitigated: response pools
   are absent; this is about retrieved transitions and true dynamics.
4. Paper could be confused with WAM score-tail audit. Mitigated: WAM v2 audits
   planner-score tails generally; this paper isolates stale retrieval
   provenance and memory dropout.
5. Paper could be confused with RSSM/Dreamer paper. Mitigated: no RSSM latent
   dynamics; mechanism is external memory support mismatch.
6. Paper could be confused with CEM/MCTS papers. Mitigated: no adaptive search
   refit/tree expansion; it is fixed candidate selection over memory-biased
   rollouts.
7. Novelty could be attacked by retrieval RL. Answer: retrieval RL improves
   control; this audits stale retrieval under selection.
8. Novelty could be attacked by model-based episodic memory. Answer: closest
   prior, but not focused on N-dependent stale-case amplification.
9. Novelty could be attacked by proxy overoptimization. Answer: yes, and this
   instantiates a memory-specific proxy/true inversion with provenance
   diagnostics.
10. Synthetic one-dimensional world is too simple. Residual risk disclosed;
    accepted as mechanism isolation.
11. Smoke-only evidence is too weak. Addressed by attempting the paper preset;
    manuscript/build should use `results/paper` when available.
12. Repair might look like an oracle. Guarded: repair uses age, disagreement,
    and dropout risk, not hidden regime labels.
13. Oracle baseline might be mistaken for deployable. Guarded: explicitly
    labeled hidden-regime oracle.
14. Claim that memory is harmful generally would be false. Guarded in abstract
    and limitations.
15. The finite mechanism might be too trivial. Guarded: paper says the
    proposition is not deep probability theory and uses it diagnostically.
16. The paper might understate related work. Literature matrix has 129 entries
    and hostile prior-work set is explicit.
17. The contribution list might sound generic. Fixed around memory-impostor
    definition, provenance diagnostics, benchmark, and repair.
18. Diagnostics could be post-hoc. Mitigated: retrieval precision, staleness,
    dropout variance, hallucinated-rollout rate, and proxy-true gap are
    mechanistically tied to the failure.
19. Hallucinated-rollout threshold could be arbitrary. Appendix says it is a
    diagnostic only, not the selector.
20. Staleness may be too constructed. Accepted: controlled support mismatch is
    the benchmark design.
21. Missing neural simulator evidence is a limitation. Explicitly scoped.
22. Missing robotics evidence is a limitation. Explicitly scoped.
23. Missing large memory stores is a limitation. Explicitly scoped.
24. Related paper side-by-side test with LLM: distinct.
25. Side-by-side test with WAM: distinct.
26. Side-by-side test with graph/token/object papers: still has a concrete
    memory provenance mechanism rather than generic selected-tail law.
27. The title avoids "Best-of-N" penalty. Fixed.
28. Abstract still says finite-N, but only as mechanism. Acceptable.
29. Body uses candidate-budget and max-score selector language where mathematically needed. Fixed.
30. README title now supports fresh-agent discovery. Fixed.
31. Build target now creates v2 Desktop artifact. Fixed.
32. The final audit no longer says the paper is not ready as-is without
    accounting for paper-preset evidence. Fixed.
33. Claims remain scoped to controlled evidence. Safe.
34. Tests cover reproducibility and mechanism. Pending final run.
35. Paper build needs log check. Pending final build.
36. PDF needs visual check. Pending final render.
37. Desktop source map needs v2 row. Pending after build.
38. GitHub push must use correct repo. Remote verified.
39. No unrelated files should be staged. Pending status check.
40. Paper preset timeout must be resolved. Pending completion or documented
    fallback.
41. If paper preset completes, macros must use it. Build script already
    prefers `results/paper`.
42. If paper preset fails, manuscript must not imply full preset. Current text
    mostly reports macro values; source doc scopes fallback.
43. The repair gap should not be oversold. It says narrowed, not eliminated.
44. The memory-impostor term is defined early. Fixed.
45. The paper remains anonymous. Existing tests cover finalcopy/style.
46. Links and style should be checked in rendered PDF. Pending.
47. Citation breadth appears adequate from existing references/literature map.
48. Empirical claims are tied to generated macros. Safe.
49. Residual risk is external validity, not duplicate-wrapper identity.
50. Final reviewer stance: v2 is submission-shaped as a scoped mechanism paper
    if the final build and checks pass.
