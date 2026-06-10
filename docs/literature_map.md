# Literature Map

This map records the first autonomous literature pass used to choose the final angle. The sweep intentionally spans model-based RL/world models, retrieval-augmented models, episodic memory, case-based reasoning, offline support-mismatch work, and Best-of-N/test-time inference.

- Total matrix entries: 129
- Serious skim set: 46 papers
- Deep/hostile close-read set: 25 papers
- Hostile prior-work set: 12 papers

## Cluster Counts

- LLM planning: 1
- LLM world models: 1
- OOD robustness: 2
- agents: 1
- agents/memory: 4
- calibration: 1
- case-based reasoning: 7
- counterfactuals: 1
- episodic RL: 9
- episodic memory: 1
- episodic world models: 4
- memory architectures: 7
- memory robustness: 3
- memory theory: 1
- model-based RL: 10
- offline RL: 5
- proxy optimization: 3
- retrieval RL: 1
- retrieval augmented models: 18
- retrieval bias: 1
- retrieval diagnostics: 4
- retrieval repair: 2
- search/planning: 2
- survey: 5
- test-time inference: 9
- trajectory models: 4
- uncertainty: 4
- world models: 18

## Closest Read Set

| title | cluster | threat | why it matters |
|---|---|---|---|
| Model-Based Episodic Memory Induces Dynamic Hybrid Controls | episodic world models | high | Very close because it combines model-based control and episodic memory; does not analyze Best-of-N amplification of stale retrieval. |
| Leveraging Episodic Memory to Improve World Models for Reinforcement Learning | episodic world models | high | Closest world-model memory prior; project differs by focusing on BoN-induced failure and repair diagnostics. |
| Retrieval Augmented Reinforcement Learning | retrieval RL | high | Strongest hostile retrieval-RL prior; does not study Best-of-N candidate selection over memory-augmented dynamics. |
| Neural Episodic Control | episodic RL | high | Closest episodic-control ancestor; stores values, not transition world-model rollouts selected by BoN. |
| Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | test-time inference | high | Strongest BoN prior; project differs by making memory retrieval the source of proxy/true inversion. |
| MERLIN: Unsupervised Predictive Memory in a Goal-Directed Agent | episodic RL | high | Strong memory-agent prior; does not isolate stale retrieval under BoN planning. |
| Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches | case-based reasoning | high | Foundational case retrieval; does not cover modern BoN world-model selection pathology. |
| Generalization through Memorization: Nearest Neighbor Language Models | retrieval augmented models | high | Very close memory interpolation mechanism; no planning/BoN true-utility analysis. |
| Scaling Laws for Reward Model Overoptimization | proxy optimization | high | Closest proxy-Goodhart theory; project instantiates architecture-specific memory retrieval mechanism. |
| Building Spatial World Models from Sparse Transitional Episodic Memories | episodic world models | high | Very close title/area; focus is building spatial maps, not BoN retrieval bias. |
| Event-Centric World Modeling with Memory-Augmented Retrieval for Decision Making | episodic world models | high | Close contemporary memory-world-model architecture; still not Best-of-N failure diagnostic. |
| Jailbreaking Large Language Models with Best-of-N | test-time inference | high | Mechanistically similar finite-N amplification; not memory-augmented dynamics. |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | retrieval augmented models | medium | Core retrieval-augmented architecture; no world-model BoN failure. |
| Retrieval-Augmented Language Model Pre-Training with Trillions of Tokens | retrieval augmented models | medium | Strong retrieval prior; not a world model or BoN memory-staleness diagnostic. |
| Self-Consistency Improves Chain of Thought Reasoning in Language Models | test-time inference | medium | Core multi-sample inference ancestor; not memory-augmented world models. |
| World Models | world models | medium | Defines the world-model setting but not memory retrieval or N-dependent selector pathology. |
| Learning Latent Dynamics for Planning from Pixels | world models | medium | Close planning-world-model ancestor; no episodic memory retrieval mechanism. |
| Dream to Control: Learning Behaviors by Latent Imagination | world models | medium | Major world-model reference; does not test external-memory Best-of-N failure. |
| Model-Free Episodic Control | episodic RL | medium | Uses episodic memory for control, but no learned world-model Best-of-N selector. |
| Episodic Reinforcement Learning with Associative Memory | episodic RL | medium | Memory retrieval for RL but not transition-model rollout selection. |
| REALM: Retrieval-Augmented Language Model Pre-Training | retrieval augmented models | medium | Retrieval model ancestor; text QA not dynamics or planning. |
| Memorizing Transformers | retrieval augmented models | medium | Inference-time memory retrieval but not world-model rollouts or selection failure. |
| Training Verifiers to Solve Math Word Problems | test-time inference | medium | Best-of-N verifier paradigm; not memory-stale dynamics. |
| Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models | test-time inference | medium | BoN-specific but text-generation focus; no episodic world-model retrieval. |
| When to Trust Your Model: Model-Based Policy Optimization | model-based RL | medium | Close on model-bias mitigation; no retrieval stale-case mechanism. |

## Takeaway

The sweep argues against claiming a new memory-augmented world-model architecture. Retrieval, episodic memory, case-based reasoning, and model-based RL are all mature threads. The viable standalone angle is narrower: Best-of-N selection can amplify a stale-retrieval error mode that is easy to miss when memory quality and candidate selection are evaluated separately.
