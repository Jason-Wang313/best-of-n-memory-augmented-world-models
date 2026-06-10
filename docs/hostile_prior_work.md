# Hostile Prior Work

These are the papers a skeptical reviewer is most likely to cite when arguing that the project is not new. The final paper therefore avoids claiming a new memory architecture and instead claims a mechanism-specific diagnostic plus a controlled repair.

1. **Retrieval Augmented Reinforcement Learning.** It already retrieves experience for RL, but its contribution is improved agent learning/control. It does not ask what a Best-of-N selector does when a retrieval-augmented transition model is stale or support mismatched.
2. **Model-Based Episodic Memory Induces Dynamic Hybrid Controls.** It is closest in combining model-based control and episodic memory. It does not isolate stale retrieved transitions as a proxy/true inversion amplified by N.
3. **Leveraging Episodic Memory to Improve World Models for Reinforcement Learning.** This is the closest title-level threat. The proposed project differs by treating memory as a possible failure source and by measuring retrieval precision, dropout variance, and BoN hallucinated rollout risk.
4. **Neural Episodic Control.** It uses differentiable dictionaries for fast value lookup. The present work uses transition memories inside a world model and evaluates generated rollouts under true dynamics.
5. **Generalization through Memorization: Nearest Neighbor Language Models.** kNN-LM is the cleanest precedent for interpolation with retrieved memories. It does not contain planning, true dynamics, latent regimes, or N-dependent selection.
6. **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.** It studies BoN/test-time compute in LLMs. The present mechanism specializes BoN failure to stale retrieval in memory-augmented world models.
7. **Scaling Laws for Reward Model Overoptimization.** Reward overoptimization covers proxy hacking broadly. Our contribution is architecture-specific: retrieval support mismatch creates the high-proxy low-true candidate type.
8. **Jailbreaking Large Language Models with Best-of-N.** It shows rare bad samples can be amplified by BoN. Our result is not about safety prompts; it formalizes and measures stale memory-impostor rollouts.
9. **Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches.** CBR anticipated retrieve/reuse/revise/retain. The new angle is finite-N selection over learned model rollouts with measurable latent support mismatch.
10. **Building Spatial World Models from Sparse Transitional Episodic Memories.** It constructs spatial maps from sparse episodes. It does not evaluate a Best-of-N planner selecting wrong futures due to stale cases.

## Reviewer-Attack Summary

- Attack: episodic memory for RL already exists. Response: yes; the paper studies a different failure mode at the interface of retrieval, learned dynamics, and BoN selection.
- Attack: proxy overoptimization already exists. Response: yes; the finite-N law is simple, and the contribution is the architecture-specific instantiation and measurement of stale memory-impostor rollouts.
- Attack: synthetic setting is too simple. Response: the paper frames the benchmark as a controlled diagnostic, not a deployed-agent proof.
