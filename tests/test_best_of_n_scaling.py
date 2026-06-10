from pathlib import Path

from memory_bon_world_models.experiment import ExperimentConfig, run_suite


def test_best_of_n_amplifies_naive_memory_impostors(tmp_path: Path):
    config = ExperimentConfig(seed=13, trials=12, ns=(1, 64), stale_fractions=(0.85,), memory_total=240)
    _, summary = run_suite(config, output_dir=tmp_path)
    data = summary.set_index(["strategy", "n"])

    naive_n1 = data.loc[("memory_bon", 1)]
    naive_n64 = data.loc[("memory_bon", 64)]
    repaired_n64 = data.loc[("repaired_memory_bon", 64)]

    assert naive_n64["proxy_score"] > naive_n1["proxy_score"]
    assert naive_n64["true_return"] < naive_n1["true_return"] - 1.0
    assert naive_n64["hallucinated_rollout"] > naive_n1["hallucinated_rollout"]
    assert repaired_n64["true_return"] > naive_n64["true_return"] + 4.0


def test_oracle_upper_bound_outperforms_naive_under_stale_shift(tmp_path: Path):
    config = ExperimentConfig(seed=21, trials=10, ns=(16,), stale_fractions=(0.93,), memory_total=240)
    _, summary = run_suite(config, output_dir=tmp_path)
    data = summary.set_index(["strategy", "n"])

    oracle = data.loc[("oracle_memory_bon", 16)]
    naive = data.loc[("memory_bon", 16)]
    assert oracle["retrieval_precision"] > 0.95
    assert naive["retrieval_precision"] < 0.4
    assert oracle["true_return"] > naive["true_return"] + 5.0
