"""Generate the literature sweep documents used by the paper."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def paper(title, year, venue, link, cluster, contribution, relevance, threat, why, read_level="sweep"):
    return {
        "title": title,
        "year": year,
        "venue/arXiv": venue,
        "link": link,
        "cluster": cluster,
        "contribution": contribution,
        "relevance score": relevance,
        "threat level": threat,
        "why it does or does not subsume this project": why,
        "read level": read_level,
    }


ENTRIES = [
    paper("Dyna, an Integrated Architecture for Learning, Planning, and Reacting", 1990, "SIGART Bulletin", "https://doi.org/10.1145/122344.122377", "model-based RL", "Introduced learned-model planning from real and simulated experience.", 6, "medium", "Foundational model-based control but no retrieval-augmented world model or Best-of-N selection failure.", "serious"),
    paper("Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time", 1993, "Machine Learning", "https://doi.org/10.1007/BF00993104", "model-based RL", "Prioritized model-based updates from stored transitions.", 5, "low", "Uses stored experience but not episodic retrieval as an inference-time dynamics component.", "sweep"),
    paper("PILCO: A Model-Based and Data-Efficient Approach to Policy Search", 2011, "ICML", "https://dl.acm.org/doi/10.5555/3104482.3104588", "model-based RL", "Gaussian-process dynamics with uncertainty-aware policy search.", 6, "medium", "Close in uncertainty spirit; lacks memory retrieval and finite-N selector amplification.", "serious"),
    paper("Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images", 2015, "NeurIPS", "https://arxiv.org/abs/1506.07365", "world models", "Latent dynamics model for control.", 4, "low", "World-model baseline lineage, not about episodic memories or selection bias.", "sweep"),
    paper("The Predictron: End-To-End Learning and Planning", 2017, "ICML", "https://arxiv.org/abs/1612.08810", "world models", "Abstract planning model trained end-to-end.", 4, "low", "Planning with learned dynamics but no retrieval memory or BoN retrieval bias.", "sweep"),
    paper("Value Prediction Network", 2017, "NeurIPS", "https://arxiv.org/abs/1707.03497", "world models", "Model-based value lookahead in abstract latent states.", 4, "low", "Relevant planning architecture; does not study external memory support mismatch.", "sweep"),
    paper("Imagination-Augmented Agents for Deep Reinforcement Learning", 2017, "NeurIPS", "https://arxiv.org/abs/1707.06203", "world models", "Agents use learned environment rollouts as imagination.", 5, "low", "Studies imagined rollouts but not retrieval or Best-of-N selection over memory-impostor candidates.", "sweep"),
    paper("World Models", 2018, "arXiv", "https://arxiv.org/abs/1803.10122", "world models", "VAE-MDN-RNN environment model used for policy learning in dreams.", 7, "medium", "Defines the world-model setting but not memory retrieval or N-dependent selector pathology.", "deep"),
    paper("Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models", 2018, "NeurIPS", "https://arxiv.org/abs/1805.12114", "model-based RL", "PETS uses probabilistic ensembles and MPC.", 6, "medium", "Close in uncertainty-aware planning, but memory is parametric/ensemble not retrieval-stale cases.", "serious"),
    paper("Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion", 2018, "NeurIPS", "https://arxiv.org/abs/1807.01675", "model-based RL", "Short model rollouts with ensembles reduce model bias.", 4, "low", "Addresses model bias but not memory retrieval or BoN amplification.", "sweep"),
    paper("Learning Latent Dynamics for Planning from Pixels", 2019, "ICML", "https://arxiv.org/abs/1811.04551", "world models", "PlaNet learns latent dynamics and plans by CEM.", 7, "medium", "Close planning-world-model ancestor; no episodic memory retrieval mechanism.", "deep"),
    paper("Dream to Control: Learning Behaviors by Latent Imagination", 2020, "ICLR", "https://arxiv.org/abs/1912.01603", "world models", "Dreamer trains actor-critic from latent imagination.", 7, "medium", "Major world-model reference; does not test external-memory Best-of-N failure.", "deep"),
    paper("Mastering Atari with Discrete World Models", 2021, "ICLR", "https://arxiv.org/abs/2010.02193", "world models", "DreamerV2 scales latent imagination to Atari.", 5, "low", "Supports world-model context; no retrieval or stale case selection.", "sweep"),
    paper("Mastering Diverse Domains through World Models", 2023, "arXiv/Nature", "https://arxiv.org/abs/2301.04104", "world models", "DreamerV3 general world-model agent across many tasks.", 6, "medium", "Hostile only as broad world-model prior; contribution is orthogonal diagnostic.", "serious"),
    paper("Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", 2020, "Nature", "https://doi.org/10.1038/s41586-020-03051-4", "world models", "MuZero learns dynamics for search without explicit environment model.", 6, "medium", "Search over learned dynamics, but not retrieval-memory bias or BoN candidate selection.", "serious"),
    paper("Mastering Atari Games with Limited Data", 2021, "NeurIPS", "https://arxiv.org/abs/2111.00210", "world models", "EfficientZero improves MuZero sample efficiency.", 4, "low", "Relevant learned-model planning lineage; no external memory retrieval.", "sweep"),
    paper("Model-Based Reinforcement Learning for Atari", 2019, "ICLR", "https://arxiv.org/abs/1903.00374", "world models", "SimPLe trains policies from generated rollouts.", 4, "low", "Deals with model-generated data; not memory-augmented inference.", "sweep"),
    paper("When to Trust Your Model: Model-Based Policy Optimization", 2019, "NeurIPS", "https://arxiv.org/abs/1906.08253", "model-based RL", "MBPO uses short model rollouts to avoid compounding error.", 6, "medium", "Close on model-bias mitigation; no retrieval stale-case mechanism.", "deep"),
    paper("Temporal Difference Learning for Model Predictive Control", 2022, "ICML", "https://arxiv.org/abs/2203.04955", "model-based RL", "TD-MPC combines latent dynamics, value learning, and MPC.", 5, "low", "Planning baseline family, not retrieval memory.", "sweep"),
    paper("TD-MPC2: Scalable, Robust World Models for Continuous Control", 2024, "ICLR", "https://arxiv.org/abs/2310.16828", "model-based RL", "Scales TD-MPC with robust latent dynamics.", 5, "low", "Modern world-model reference but not episodic retrieval failure.", "sweep"),
    paper("Trajectory Transformer: Reinforcement Learning as Sequence Modeling", 2021, "NeurIPS", "https://arxiv.org/abs/2106.02039", "trajectory models", "Plans by beam search over learned trajectory sequences.", 5, "medium", "Selection/search over candidate futures is relevant; no external memory retrieval support mismatch.", "serious"),
    paper("Planning with Diffusion for Flexible Behavior Synthesis", 2022, "ICML", "https://arxiv.org/abs/2205.09991", "trajectory models", "Diffuser uses diffusion models for trajectory planning.", 4, "low", "Generative planning context, not memory-augmented retrieval.", "sweep"),
    paper("Decision Transformer: Reinforcement Learning via Sequence Modeling", 2021, "NeurIPS", "https://arxiv.org/abs/2106.01345", "trajectory models", "Offline RL as conditional sequence modeling.", 4, "low", "Retrieves only through parametric sequence model; no memory-store diagnostic.", "sweep"),
    paper("UniPi: Learning Universal Policies via Text-Guided Video Generation", 2023, "NeurIPS", "https://arxiv.org/abs/2302.00111", "trajectory models", "Video generation as policy planning.", 3, "low", "Candidate future generation context only.", "sweep"),
    paper("Genie: Generative Interactive Environments", 2024, "arXiv", "https://arxiv.org/abs/2402.15391", "world models", "Foundation world model for interactive environments.", 4, "low", "Broad motivation for future-world sampling but no memory retrieval analysis.", "sweep"),
    paper("IRIS: Transformers are Sample-Efficient World Models", 2022, "ICLR", "https://arxiv.org/abs/2209.00588", "world models", "Transformer world model for Atari.", 4, "low", "Parametric world model lineage only.", "sweep"),
    paper("OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving", 2024, "arXiv", "https://arxiv.org/abs/2311.16038", "world models", "Occupancy forecasting world model for driving.", 4, "low", "World-model domain motivation; no retrieval or BoN failure.", "sweep"),
    paper("Neural Turing Machines", 2014, "arXiv", "https://arxiv.org/abs/1410.5401", "memory architectures", "Differentiable external memory for neural networks.", 6, "medium", "Foundational memory architecture; does not study retrieval-stale world-model planning.", "serious"),
    paper("End-To-End Memory Networks", 2015, "NeurIPS", "https://arxiv.org/abs/1503.08895", "memory architectures", "Memory network trained end-to-end for QA/reasoning.", 4, "low", "Memory-augmented neural architecture but not dynamics or BoN.", "sweep"),
    paper("Differentiable Neural Computers", 2016, "Nature", "https://www.nature.com/articles/nature20101", "memory architectures", "Neural controller with differentiable read/write memory.", 6, "medium", "External memory ancestor but lacks retrieval case staleness and planning selection.", "serious"),
    paper("One-shot Learning with Memory-Augmented Neural Networks", 2016, "ICML", "https://arxiv.org/abs/1605.06065", "memory architectures", "Memory-augmented meta-learning for one-shot classification.", 4, "low", "Memory motivation, not world models or Best-of-N inference.", "sweep"),
    paper("Neural Episodic Control", 2017, "ICML", "https://proceedings.mlr.press/v70/pritzel17a.html", "episodic RL", "Differentiable neural dictionary for fast value lookup.", 9, "high", "Closest episodic-control ancestor; stores values, not transition world-model rollouts selected by BoN.", "hostile"),
    paper("Model-Free Episodic Control", 2016, "arXiv", "https://arxiv.org/abs/1606.04460", "episodic RL", "Episodic non-parametric control from high-return experiences.", 7, "medium", "Uses episodic memory for control, but no learned world-model Best-of-N selector.", "deep"),
    paper("MERLIN: Unsupervised Predictive Memory in a Goal-Directed Agent", 2018, "ICML", "https://arxiv.org/abs/1803.10760", "episodic RL", "Memory-based agent with predictive modeling.", 8, "high", "Strong memory-agent prior; does not isolate stale retrieval under BoN planning.", "hostile"),
    paper("Episodic Reinforcement Learning with Associative Memory", 2020, "ICLR", "https://openreview.net/forum?id=HkxjqxBYDB", "episodic RL", "Associative memory improves episodic RL.", 7, "medium", "Memory retrieval for RL but not transition-model rollout selection.", "deep"),
    paper("Solving Continuous Control with Episodic Memory", 2021, "IJCAI", "https://www.ijcai.org/proceedings/2021/0365.pdf", "episodic RL", "Episodic memory for continuous control.", 6, "medium", "Episodic-control comparison; not support-mismatch BoN world models.", "serious"),
    paper("Model-Based Episodic Memory Induces Dynamic Hybrid Controls", 2021, "NeurIPS", "https://proceedings.neurips.cc/paper/2021/hash/fe73f687e5bc5280214e0486b273a5f9-Abstract.html", "episodic world models", "Model-based episodic memory of trajectories for control.", 10, "high", "Very close because it combines model-based control and episodic memory; does not analyze Best-of-N amplification of stale retrieval.", "hostile"),
    paper("Leveraging Episodic Memory to Improve World Models for Reinforcement Learning", 2022, "MEMARI workshop", "https://memari-workshop.github.io/papers/paper_3.pdf", "episodic world models", "Episodic memory module for world-model transitions.", 10, "high", "Closest world-model memory prior; project differs by focusing on BoN-induced failure and repair diagnostics.", "hostile"),
    paper("Retrieval Augmented Reinforcement Learning", 2022, "ICML", "https://proceedings.mlr.press/v162/goyal22a.html", "retrieval RL", "Retrieval over past experience improves RL agents.", 10, "high", "Strongest hostile retrieval-RL prior; does not study Best-of-N candidate selection over memory-augmented dynamics.", "hostile"),
    paper("Neural Map: Structured Memory for Deep Reinforcement Learning", 2018, "ICLR", "https://openreview.net/forum?id=Bk9zbyZCZ", "memory architectures", "Spatial memory for navigation agents.", 5, "low", "Structured memory for agents, not retrieval case bias in model rollouts.", "sweep"),
    paper("Memory Augmented Control Networks", 2016, "ICLR", "https://arxiv.org/abs/1511.06295", "memory architectures", "Memory helps control policies store task information.", 5, "low", "Control with memory, but no world-model retrieval or BoN.", "sweep"),
    paper("Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches", 1994, "AI Communications", "https://doi.org/10.3233/AIC-1994-7104", "case-based reasoning", "Classic CBR framework of retrieve, reuse, revise, retain.", 8, "high", "Foundational case retrieval; does not cover modern BoN world-model selection pathology.", "hostile"),
    paper("Case-Based Reasoning", 1993, "Morgan Kaufmann", "https://www.sciencedirect.com/book/9781558602373/case-based-reasoning", "case-based reasoning", "Kolodner book formalizing episodic case-based problem solving.", 6, "medium", "General CBR prior; project is a specific finite-N failure mode.", "serious"),
    paper("Case-Based Reasoning: A Review", 1994, "Knowledge Engineering Review", "https://doi.org/10.1017/S0269888900007098", "case-based reasoning", "Survey of CBR systems and memory organization.", 5, "medium", "Historical review, not neural world-model planning.", "sweep"),
    paper("Qualitative Case-Based Reasoning and Learning", 2018, "Artificial Intelligence Review", "https://doi.org/10.1007/s10462-017-9537-8", "case-based reasoning", "CBR with qualitative spatial representations and RL links.", 6, "medium", "Case-based RL relation; no BoN selector amplification.", "serious"),
    paper("REALM: Retrieval-Augmented Language Model Pre-Training", 2020, "ICML", "https://arxiv.org/abs/2002.08909", "retrieval augmented models", "End-to-end pretraining with latent document retrieval.", 7, "medium", "Retrieval model ancestor; text QA not dynamics or planning.", "deep"),
    paper("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020, "NeurIPS", "https://arxiv.org/abs/2005.11401", "retrieval augmented models", "RAG combines parametric generator with non-parametric document memory.", 8, "medium", "Core retrieval-augmented architecture; no world-model BoN failure.", "deep"),
    paper("Dense Passage Retrieval for Open-Domain Question Answering", 2020, "EMNLP", "https://arxiv.org/abs/2004.04906", "retrieval augmented models", "Dense retriever for open-domain QA.", 4, "low", "Retriever foundation only.", "sweep"),
    paper("Retrieval-Augmented Language Model Pre-Training with Trillions of Tokens", 2022, "ICML", "https://arxiv.org/abs/2112.04426", "retrieval augmented models", "RETRO scales retrieval-augmented LMs.", 8, "medium", "Strong retrieval prior; not a world model or BoN memory-staleness diagnostic.", "deep"),
    paper("Generalization through Memorization: Nearest Neighbor Language Models", 2020, "ICLR", "https://arxiv.org/abs/1911.00172", "retrieval augmented models", "kNN-LM interpolates neural predictions with datastore retrieval.", 8, "high", "Very close memory interpolation mechanism; no planning/BoN true-utility analysis.", "hostile"),
    paper("Memorizing Transformers", 2022, "ICLR", "https://arxiv.org/abs/2203.08913", "retrieval augmented models", "Transformer accesses large kNN memory at inference.", 7, "medium", "Inference-time memory retrieval but not world-model rollouts or selection failure.", "deep"),
    paper("Few-shot Question Answering by Pretraining Span Selection", 2019, "ACL", "https://arxiv.org/abs/1901.07054", "retrieval augmented models", "Retrieval plus span selection for QA.", 3, "low", "Retrieval QA background only.", "sweep"),
    paper("Fusion-in-Decoder: Leveraging Passage Retrieval with Generative Models", 2021, "EACL", "https://arxiv.org/abs/2007.01282", "retrieval augmented models", "FiD combines many retrieved passages in decoder.", 4, "low", "RAG architecture background, not dynamics.", "sweep"),
    paper("Atlas: Few-shot Learning with Retrieval Augmented Language Models", 2022, "JMLR", "https://arxiv.org/abs/2208.03299", "retrieval augmented models", "Retrieval-augmented LM scales few-shot learning.", 6, "medium", "Retrieval augmentation close architecturally but not world-model planning.", "serious"),
    paper("REPLUG: Retrieval-Augmented Black-Box Language Models", 2023, "arXiv", "https://arxiv.org/abs/2301.12652", "retrieval augmented models", "Retrieval improves frozen black-box LMs.", 4, "low", "Retrieval at inference, no dynamics or BoN failure.", "sweep"),
    paper("Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", 2023, "ICLR", "https://arxiv.org/abs/2310.11511", "retrieval augmented models", "Controls retrieval and critique tokens.", 5, "medium", "Includes retrieval gating ideas; not transition-memory BoN.", "serious"),
    paper("FLARE: Forward-Looking Active Retrieval Augmented Generation", 2023, "arXiv", "https://arxiv.org/abs/2305.06983", "retrieval augmented models", "Actively retrieves based on generation uncertainty.", 5, "medium", "Retrieval timing/gating cousin, not world-model Best-of-N.", "sweep"),
    paper("In-Context Retrieval-Augmented Language Models", 2024, "TACL", "https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00605/118118/In-Context-Retrieval-Augmented-Language-Models", "retrieval augmented models", "In-context examples as retrieval augmentation for LMs.", 5, "low", "Retrieval-as-context only.", "sweep"),
    paper("KATE: KNN-Augmented In-Context Example Selection", 2022, "NAACL", "https://arxiv.org/abs/2205.13033", "retrieval augmented models", "Nearest-neighbor demonstration selection.", 4, "low", "Selection via retrieval but no world dynamics.", "sweep"),
    paper("CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs", 2024, "arXiv", "https://arxiv.org/abs/2404.04302", "case-based reasoning", "Combines CBR cycle and RAG for LLMs.", 6, "medium", "CBR/RAG synthesis; no model-based RL or BoN selector failure.", "serious"),
    paper("Retrieval Meets Long Context Large Language Models", 2024, "arXiv", "https://arxiv.org/abs/2310.03025", "retrieval augmented models", "Compares retrieval with long-context models.", 3, "low", "Retrieval systems context only.", "sweep"),
    paper("Lost in the Middle: How Language Models Use Long Contexts", 2024, "TACL", "https://arxiv.org/abs/2307.03172", "retrieval bias", "Shows position-dependent use of retrieved context.", 5, "medium", "A retrieval failure diagnostic, but unrelated to dynamics/BoN.", "sweep"),
    paper("Corrective Retrieval Augmented Generation", 2024, "arXiv", "https://arxiv.org/abs/2401.15884", "retrieval repair", "Adds correction mechanisms for low-quality retrieval.", 5, "medium", "Repair inspiration; not world-model memory staleness under selection.", "sweep"),
    paper("Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models", 2024, "NAACL", "https://arxiv.org/abs/2403.14403", "retrieval repair", "Adaptive retrieval complexity for QA.", 4, "low", "Gating inspiration only.", "sweep"),
    paper("RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023, "arXiv", "https://arxiv.org/abs/2309.15217", "retrieval diagnostics", "Metrics for RAG faithfulness and context quality.", 4, "low", "Diagnostic framing but text QA not dynamics.", "sweep"),
    paper("Benchmarking Large Language Models in Retrieval-Augmented Generation", 2023, "arXiv", "https://arxiv.org/abs/2309.01431", "retrieval diagnostics", "RGB benchmark for RAG robustness.", 4, "low", "Retrieval robustness context; no BoN planning.", "sweep"),
    paper("Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022, "ICLR", "https://arxiv.org/abs/2203.11171", "test-time inference", "Samples multiple reasoning paths and marginalizes answers.", 8, "medium", "Core multi-sample inference ancestor; not memory-augmented world models.", "deep"),
    paper("Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023, "NeurIPS", "https://arxiv.org/abs/2305.10601", "test-time inference", "Searches over intermediate thoughts with evaluator.", 6, "medium", "Search/evaluator setting; no external transition memory.", "serious"),
    paper("Graph of Thoughts: Solving Elaborate Problems with Large Language Models", 2023, "AAAI", "https://arxiv.org/abs/2308.09687", "test-time inference", "Graph-structured inference-time reasoning.", 4, "low", "Search context, not retrieval world models.", "sweep"),
    paper("Training Verifiers to Solve Math Word Problems", 2021, "arXiv", "https://arxiv.org/abs/2110.14168", "test-time inference", "Verifier ranks sampled solutions.", 7, "medium", "Best-of-N verifier paradigm; not memory-stale dynamics.", "deep"),
    paper("Let's Verify Step by Step", 2023, "arXiv", "https://arxiv.org/abs/2305.20050", "test-time inference", "Process reward models improve verifier-guided reasoning.", 6, "medium", "Verifier search ancestor; no memory world model.", "serious"),
    paper("Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", 2024, "arXiv", "https://arxiv.org/abs/2408.03314", "test-time inference", "Analyzes Best-of-N and verifier-guided test-time compute.", 9, "high", "Strongest BoN prior; project differs by making memory retrieval the source of proxy/true inversion.", "hostile"),
    paper("Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models", 2024, "arXiv", "https://arxiv.org/abs/2412.15287", "test-time inference", "Optimizes models for BoN inference.", 7, "medium", "BoN-specific but text-generation focus; no episodic world-model retrieval.", "deep"),
    paper("Jailbreaking Large Language Models with Best-of-N", 2024, "arXiv", "https://arxiv.org/abs/2412.03556", "test-time inference", "Shows BoN can amplify rare unsafe completions.", 7, "high", "Mechanistically similar finite-N amplification; not memory-augmented dynamics.", "hostile"),
    paper("Scaling Laws for Reward Model Overoptimization", 2023, "ICML", "https://arxiv.org/abs/2210.10760", "proxy optimization", "Quantifies reward-model overoptimization.", 8, "high", "Closest proxy-Goodhart theory; project instantiates architecture-specific memory retrieval mechanism.", "hostile"),
    paper("Learning to Summarize from Human Feedback", 2020, "NeurIPS", "https://arxiv.org/abs/2009.01325", "proxy optimization", "RLHF and reward-model ranking for summaries.", 4, "low", "Ranking/proxy background only.", "sweep"),
    paper("WebGPT: Browser-Assisted Question-Answering with Human Feedback", 2021, "arXiv", "https://arxiv.org/abs/2112.09332", "test-time inference", "Uses browsing, demonstrations, and human preference ranking.", 4, "low", "Candidate ranking context, not world-model memory.", "sweep"),
    paper("Constitutional AI: Harmlessness from AI Feedback", 2022, "arXiv", "https://arxiv.org/abs/2212.08073", "proxy optimization", "AI feedback and preference optimization.", 3, "low", "Proxy optimization context only.", "sweep"),
    paper("AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", 2018, "Science", "https://doi.org/10.1126/science.aar6404", "search/planning", "MCTS with learned policy/value.", 4, "low", "Search lineage; not learned external memory dynamics.", "sweep"),
    paper("Monte-Carlo Tree Search and Rapid Action Value Estimation in Computer Go", 2011, "AI", "https://doi.org/10.1016/j.artint.2010.11.007", "search/planning", "MCTS/RAVE search methods.", 3, "low", "Search context only.", "sweep"),
    paper("Planning with Learned Object Importance in Model-Based Reinforcement Learning", 2020, "NeurIPS", "https://arxiv.org/abs/2007.08341", "model-based RL", "Object-centric model-based planning.", 3, "low", "World-model architecture background only.", "sweep"),
    paper("A Survey on Model-Based Reinforcement Learning", 2022, "arXiv", "https://arxiv.org/abs/2206.09328", "survey", "Surveys model-based RL methods.", 5, "medium", "Broad coverage confirms gap at memory retrieval plus BoN selection.", "serious"),
    paper("Retrieval-Augmented Generation for AI-Generated Content: A Survey", 2024, "arXiv", "https://arxiv.org/abs/2402.19473", "survey", "RAG survey across methods and applications.", 5, "medium", "Surveys retrieval but does not cover world-model BoN dynamics.", "sweep"),
    paper("A Survey of Large Language Models for Test-Time Compute", 2025, "arXiv", "https://arxiv.org/abs/2501.02497", "survey", "Survey of inference-time scaling methods.", 4, "low", "BoN context only; no retrieval world-model focus.", "sweep"),
    paper("A Survey of Episodic Memory in Reinforcement Learning", 2018, "Frontiers/PMC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5953519/", "survey", "Reviews RL and episodic memory connections in humans and animals.", 5, "medium", "Motivates memory but not technical BoN failure.", "sweep"),
    paper("Relating Hopfield Networks to Episodic Control", 2024, "NeurIPS", "https://openreview.net/forum?id=59DmXSBG6S", "episodic RL", "Connects DND episodic control to Hopfield networks.", 5, "medium", "Theoretical memory prior; not transition-planning BoN.", "sweep"),
    paper("Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models", 2023, "arXiv", "https://arxiv.org/abs/2308.00044", "memory theory", "General associative-memory formulation.", 3, "low", "Associative memory theory background only.", "sweep"),
    paper("Episodic Memory Reader: Learning What to Remember for Question Answering from Streaming Data", 2016, "ACL", "https://arxiv.org/abs/1603.07391", "memory architectures", "Learns to store and read episodic memories for QA.", 4, "low", "Memory reader background; no world model.", "sweep"),
    paper("Fast Reinforcement Learning with Generalized Policy Updates", 2018, "PNAS", "https://www.pnas.org/doi/10.1073/pnas.1907370116", "episodic RL", "Successor features enable transfer using stored policies.", 4, "low", "Transfer/control memory context, not retrieval dynamics.", "sweep"),
    paper("Temporally Extended Successor Feature Neural Episodic Control", 2024, "PMC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11219751/", "episodic RL", "Extends episodic control with successor features.", 4, "low", "Episodic-control variant, not BoN world models.", "sweep"),
    paper("Episodic Reinforcement Learning with Expanded State-Reward Space", 2024, "arXiv", "https://arxiv.org/abs/2401.10516", "episodic RL", "Episodic control with expanded state-reward memories.", 4, "low", "Memory RL variant but no selection amplification.", "sweep"),
    paper("Building Spatial World Models from Sparse Transitional Episodic Memories", 2025, "arXiv", "https://arxiv.org/abs/2505.13696", "episodic world models", "Constructs spatial maps from sparse episodic memories.", 8, "high", "Very close title/area; focus is building spatial maps, not BoN retrieval bias.", "hostile"),
    paper("Event-Centric World Modeling with Memory-Augmented Retrieval for Decision Making", 2026, "arXiv", "https://arxiv.org/abs/2604.07392", "episodic world models", "Event memory retrieval for interpretable decision making.", 8, "high", "Close contemporary memory-world-model architecture; still not Best-of-N failure diagnostic.", "hostile"),
    paper("LLM-Based World Models Can Make Decisions Solely, But Rigorous Evaluation is Needed", 2024, "arXiv", "https://arxiv.org/abs/2411.08794", "world models", "Evaluates LLMs as world models for decision making.", 5, "medium", "World model evaluation context; not retrieval staleness.", "serious"),
    paper("Reasoning via Planning: Language Models as World Models", 2023, "ACL", "https://arxiv.org/abs/2305.14992", "LLM world models", "Uses LLM simulations for reasoning/planning.", 5, "medium", "World-model plus search cousin; no episodic memory retrieval.", "serious"),
    paper("Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents", 2022, "ICML", "https://arxiv.org/abs/2201.07207", "LLM planning", "LLMs produce plans for embodied agents.", 3, "low", "Planning context only.", "sweep"),
    paper("ReAct: Synergizing Reasoning and Acting in Language Models", 2023, "ICLR", "https://arxiv.org/abs/2210.03629", "agents", "Interleaves reasoning and acting with tools.", 3, "low", "Agent test-time loop context, not memory world models.", "sweep"),
    paper("Reflexion: Language Agents with Verbal Reinforcement Learning", 2023, "NeurIPS", "https://arxiv.org/abs/2303.11366", "agents/memory", "Agents store verbal reflections as memory.", 4, "medium", "Episodic agent memory but not dynamics-model retrieval.", "sweep"),
    paper("Voyager: An Open-Ended Embodied Agent with Large Language Models", 2023, "TMLR", "https://arxiv.org/abs/2305.16291", "agents/memory", "LLM agent builds skill library and memory.", 4, "low", "Agent memory context only.", "sweep"),
    paper("Generative Agents: Interactive Simulacra of Human Behavior", 2023, "UIST", "https://arxiv.org/abs/2304.03442", "agents/memory", "Agents retrieve and reflect on memories.", 5, "medium", "Memory retrieval and planning, but not learned transition world models or BoN.", "sweep"),
    paper("MemoryBank: Enhancing Large Language Models with Long-Term Memory", 2023, "AAAI", "https://arxiv.org/abs/2305.10250", "agents/memory", "Long-term memory for LLM conversations.", 3, "low", "Memory architecture background.", "sweep"),
    paper("Generative Semantic Workspaces for Episodic Memory", 2025, "arXiv", "https://arxiv.org/abs/2501.13121", "episodic memory", "Synthetic episodic-memory benchmark for LLMs.", 4, "low", "Episodic-memory benchmark but not world-model planning.", "sweep"),
    paper("Case-Based Reasoning Augmented Large Language Model for Autonomous Driving Decision Making", 2025, "arXiv", "https://arxiv.org/abs/2506.20531", "case-based reasoning", "CBR-LLM for driving maneuvers.", 5, "medium", "Case retrieval for decision making; no BoN dynamics failure.", "sweep"),
    paper("A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications", 2025, "arXiv", "https://arxiv.org/abs/2501.05030", "case-based reasoning", "Multimodal CBR-RAG framework.", 4, "low", "CBR/RAG architecture; no world model selection.", "sweep"),
    paper("Counterfactual Data Augmentation using Locally Factored Dynamics", 2020, "NeurIPS", "https://arxiv.org/abs/2007.02863", "counterfactuals", "Uses dynamics factorization for counterfactual data.", 3, "low", "Counterfactual repair inspiration only.", "sweep"),
    paper("Invariant Risk Minimization", 2020, "arXiv", "https://arxiv.org/abs/1907.02893", "OOD robustness", "Learns invariant predictors across environments.", 4, "low", "Regime-shift robustness context but not memory retrieval.", "sweep"),
    paper("Distributionally Robust Neural Networks", 2018, "ICLR", "https://openreview.net/forum?id=rybFq1-Rb", "OOD robustness", "DRO training for robustness.", 3, "low", "Robustness background only.", "sweep"),
    paper("On Calibration of Modern Neural Networks", 2017, "ICML", "https://arxiv.org/abs/1706.04599", "calibration", "Shows neural networks are miscalibrated; temperature scaling.", 4, "low", "Calibration inspiration for uncertainty gating.", "sweep"),
    paper("Deep Ensembles: A Loss Landscape Perspective", 2020, "arXiv", "https://arxiv.org/abs/1912.02757", "uncertainty", "Explains deep ensemble uncertainty behavior.", 3, "low", "Dropout/uncertainty inspiration only.", "sweep"),
    paper("Dropout as a Bayesian Approximation", 2016, "ICML", "https://arxiv.org/abs/1506.02142", "uncertainty", "MC dropout as approximate Bayesian inference.", 4, "low", "Counterfactual memory dropout is analogous but applied to retrieval cases.", "sweep"),
    paper("Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", 2017, "NeurIPS", "https://arxiv.org/abs/1612.01474", "uncertainty", "Deep ensembles for uncertainty and OOD.", 5, "low", "Uncertainty repair inspiration, not memory retrieval.", "sweep"),
    paper("Model-Based Reinforcement Learning via Meta-Policy Optimization", 2018, "CoRL", "https://arxiv.org/abs/1809.05214", "model-based RL", "Meta-policy optimization over learned dynamics.", 3, "low", "Model-based planning context only.", "sweep"),
    paper("Hallucinated Replay for Continual Reinforcement Learning", 2021, "arXiv", "https://arxiv.org/abs/2106.02626", "memory robustness", "Replay/hallucination for continual RL.", 3, "low", "Continual memory context but not BoN selection.", "sweep"),
    paper("Continual Learning with Deep Generative Replay", 2017, "NeurIPS", "https://arxiv.org/abs/1705.08690", "memory robustness", "Generative replay for continual learning.", 3, "low", "Staleness/continual learning background.", "sweep"),
    paper("Experience Replay for Continual Learning", 2019, "NeurIPS", "https://arxiv.org/abs/1811.11682", "memory robustness", "Replay buffer methods for continual learning.", 3, "low", "Replay buffer context, not inference-time retrieval.", "sweep"),
    paper("Conservative Q-Learning for Offline Reinforcement Learning", 2020, "NeurIPS", "https://arxiv.org/abs/2006.04779", "offline RL", "Penalizes out-of-distribution actions.", 4, "medium", "Support-mismatch analogy; no memory world-model BoN.", "sweep"),
    paper("Offline Reinforcement Learning: Tutorial, Review, and Perspectives", 2020, "arXiv", "https://arxiv.org/abs/2005.01643", "offline RL", "Survey of offline RL distribution shift.", 4, "medium", "Support mismatch background only.", "sweep"),
    paper("MOPO: Model-Based Offline Policy Optimization", 2020, "NeurIPS", "https://arxiv.org/abs/2005.13239", "offline RL", "Uncertainty-penalized model rollouts for offline RL.", 6, "medium", "Close repair spirit; no retrieval-stale BoN diagnostic.", "serious"),
    paper("MOReL: Model-Based Offline Reinforcement Learning", 2020, "NeurIPS", "https://arxiv.org/abs/2005.05951", "offline RL", "Pessimistic MDP for unknown states.", 5, "medium", "Pessimism under support mismatch is related but not memory retrieval.", "serious"),
    paper("COMBO: Conservative Offline Model-Based Policy Optimization", 2021, "NeurIPS", "https://arxiv.org/abs/2102.08363", "offline RL", "Combines model-based rollouts with conservative value learning.", 4, "low", "OOD model bias background.", "sweep"),
    paper("A Survey of Uncertainty in Deep Reinforcement Learning", 2020, "arXiv", "https://arxiv.org/abs/2002.11395", "uncertainty", "Reviews uncertainty estimation in deep RL.", 3, "low", "Uncertainty context only.", "sweep"),
    paper("RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems", 2024, "arXiv", "https://arxiv.org/abs/2403.09040", "retrieval diagnostics", "RAG design/evaluation framework.", 3, "low", "RAG design context; not dynamics.", "sweep"),
    paper("ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems", 2023, "arXiv", "https://arxiv.org/abs/2311.09476", "retrieval diagnostics", "Automated RAG evaluation with synthetic labels.", 3, "low", "Evaluation background.", "sweep"),
    paper("Learning to Retrieve Passages without Supervision", 2019, "arXiv", "https://arxiv.org/abs/1909.04425", "retrieval augmented models", "Unsupervised dense retrieval.", 3, "low", "Retriever training background.", "sweep"),
    paper("Nearest Neighbor Machine Translation", 2021, "ICLR", "https://arxiv.org/abs/2010.00710", "retrieval augmented models", "kNN datastore improves neural MT.", 5, "medium", "Close kNN interpolation mechanism outside world models.", "sweep"),
    paper("Retrieval-Augmented Multimodal Language Modeling", 2023, "ICML", "https://arxiv.org/abs/2211.12561", "retrieval augmented models", "Multimodal retrieval augmentation.", 3, "low", "Retrieval across modalities only.", "sweep"),
    paper("Model Predictive Control with Learned Dynamics: A Survey", 2020, "arXiv", "https://arxiv.org/abs/2012.10638", "survey", "Survey of learned-dynamics MPC.", 4, "low", "MPC context; no memory retrieval.", "sweep"),
    paper("Planning to Explore via Self-Supervised World Models", 2020, "ICML", "https://arxiv.org/abs/2005.05960", "world models", "Exploration by disagreement in learned world models.", 4, "low", "Uncertainty/world-model context.", "sweep"),
    paper("Learning and Querying Fast Generative Models for Reinforcement Learning", 2020, "arXiv", "https://arxiv.org/abs/2006.09647", "world models", "Fast learned models for RL rollouts.", 3, "low", "World-model rollout background.", "sweep"),
]


HOSTILE_NOTES = [
    ("Retrieval Augmented Reinforcement Learning", "It already retrieves experience for RL, but its contribution is improved agent learning/control. It does not ask what a Best-of-N selector does when a retrieval-augmented transition model is stale or support mismatched."),
    ("Model-Based Episodic Memory Induces Dynamic Hybrid Controls", "It is closest in combining model-based control and episodic memory. It does not isolate stale retrieved transitions as a proxy/true inversion amplified by N."),
    ("Leveraging Episodic Memory to Improve World Models for Reinforcement Learning", "This is the closest title-level threat. The proposed project differs by treating memory as a possible failure source and by measuring retrieval precision, dropout variance, and BoN hallucinated rollout risk."),
    ("Neural Episodic Control", "It uses differentiable dictionaries for fast value lookup. The present work uses transition memories inside a world model and evaluates generated rollouts under true dynamics."),
    ("Generalization through Memorization: Nearest Neighbor Language Models", "kNN-LM is the cleanest precedent for interpolation with retrieved memories. It does not contain planning, true dynamics, latent regimes, or N-dependent selection."),
    ("Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", "It studies BoN/test-time compute in LLMs. The present mechanism specializes BoN failure to stale retrieval in memory-augmented world models."),
    ("Scaling Laws for Reward Model Overoptimization", "Reward overoptimization covers proxy hacking broadly. Our contribution is architecture-specific: retrieval support mismatch creates the high-proxy low-true candidate type."),
    ("Jailbreaking Large Language Models with Best-of-N", "It shows rare bad samples can be amplified by BoN. Our result is not about safety prompts; it formalizes and measures stale memory-impostor rollouts."),
    ("Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches", "CBR anticipated retrieve/reuse/revise/retain. The new angle is finite-N selection over learned model rollouts with measurable latent support mismatch."),
    ("Building Spatial World Models from Sparse Transitional Episodic Memories", "It constructs spatial maps from sparse episodes. It does not evaluate a Best-of-N planner selecting wrong futures due to stale cases."),
]


def write_matrix() -> pd.DataFrame:
    DOCS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ENTRIES)
    if len(df) < 100:
        raise RuntimeError(f"expected at least 100 entries, found {len(df)}")
    required = [
        "title",
        "year",
        "venue/arXiv",
        "link",
        "cluster",
        "contribution",
        "relevance score",
        "threat level",
        "why it does or does not subsume this project",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")
    df.to_csv(DOCS / "related_work_matrix.csv", index=False)
    return df


def write_literature_map(df: pd.DataFrame) -> None:
    counts = Counter(df["cluster"])
    levels = Counter(df["read level"])
    deep = df[df["read level"].isin(["deep", "hostile"])].sort_values(
        ["threat level", "relevance score"], ascending=[True, False]
    )
    serious = df[df["read level"].isin(["serious", "deep", "hostile"])]

    lines = [
        "# Literature Map",
        "",
        "This map records the first autonomous literature pass used to choose the final angle. The sweep intentionally spans model-based RL/world models, retrieval-augmented models, episodic memory, case-based reasoning, offline support-mismatch work, and Best-of-N/test-time inference.",
        "",
        f"- Total matrix entries: {len(df)}",
        f"- Serious skim set: {len(serious)} papers",
        f"- Deep/hostile close-read set: {len(deep)} papers",
        f"- Hostile prior-work set: {levels.get('hostile', 0)} papers",
        "",
        "## Cluster Counts",
        "",
    ]
    for cluster, count in sorted(counts.items()):
        lines.append(f"- {cluster}: {count}")

    lines += [
        "",
        "## Closest Read Set",
        "",
        "| title | cluster | threat | why it matters |",
        "|---|---|---|---|",
    ]
    for _, row in deep.head(25).iterrows():
        lines.append(
            f"| {row['title']} | {row['cluster']} | {row['threat level']} | {row['why it does or does not subsume this project']} |"
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "The sweep argues against claiming a new memory-augmented world-model architecture. Retrieval, episodic memory, case-based reasoning, and model-based RL are all mature threads. The viable standalone angle is narrower: Best-of-N selection can amplify a stale-retrieval error mode that is easy to miss when memory quality and candidate selection are evaluated separately.",
    ]
    (DOCS / "literature_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hostile_prior_work() -> None:
    lines = [
        "# Hostile Prior Work",
        "",
        "These are the papers a skeptical reviewer is most likely to cite when arguing that the project is not new. The final paper therefore avoids claiming a new memory architecture and instead claims a mechanism-specific diagnostic plus a controlled repair.",
        "",
    ]
    for index, (title, note) in enumerate(HOSTILE_NOTES, start=1):
        lines.append(f"{index}. **{title}.** {note}")
    lines += [
        "",
        "## Reviewer-Attack Summary",
        "",
        "- Attack: episodic memory for RL already exists. Response: yes; the paper studies a different failure mode at the interface of retrieval, learned dynamics, and BoN selection.",
        "- Attack: proxy overoptimization already exists. Response: yes; the finite-N law is simple, and the contribution is the architecture-specific instantiation and measurement of stale memory-impostor rollouts.",
        "- Attack: synthetic setting is too simple. Response: the paper frames the benchmark as a controlled diagnostic, not a deployed-agent proof.",
    ]
    (DOCS / "hostile_prior_work.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_novelty_decision() -> None:
    text = """# Novelty Decision

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
"""
    (DOCS / "novelty_decision.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = write_matrix()
    write_literature_map(df)
    write_hostile_prior_work()
    write_novelty_decision()
    print(f"wrote {len(df)} related-work entries to {DOCS / 'related_work_matrix.csv'}")


if __name__ == "__main__":
    main()
