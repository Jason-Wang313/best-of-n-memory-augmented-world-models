from pathlib import Path

import pandas as pd

from memory_bon_world_models.experiment import ExperimentConfig, run_suite


def test_suite_is_reproducible_for_fixed_seed(tmp_path: Path):
    config = ExperimentConfig(seed=99, trials=4, ns=(1, 8), stale_fractions=(0.65,), memory_total=180)
    _, first = run_suite(config, output_dir=tmp_path / "a")
    _, second = run_suite(config, output_dir=tmp_path / "b")

    pd.testing.assert_frame_equal(first, second)
