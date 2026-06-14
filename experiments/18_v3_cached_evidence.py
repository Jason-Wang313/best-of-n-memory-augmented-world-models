from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v3_cached_evidence"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"

LABELS = {
    "base_bon": "non-memory",
    "memory_bon": "naive memory",
    "repaired_memory_bon": "repaired memory",
    "oracle_memory_bon": "oracle memory",
}


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = ROOT / "results" / "v3_base" / "raw_rollouts.csv"
    summary_path = ROOT / "results" / "v3_base" / "summary.csv"
    if not raw_path.exists() or not summary_path.exists():
        raise FileNotFoundError("missing results/v3_base artifacts")
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    return raw, summary


def bootstrap_ci(raw: pd.DataFrame, *, seed: int = 123, draws: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    metrics = [
        "true_return",
        "proxy_score",
        "proxy_true_gap",
        "hallucinated_rollout",
        "retrieval_precision",
        "retrieval_stale_rate",
        "diagnostic_risk",
    ]
    for (stale, strategy, n), group in raw.groupby(["stale_fraction", "strategy", "n"]):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
            rows.append(
                {
                    "stale_fraction": float(stale),
                    "strategy": strategy,
                    "n": int(n),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def budget_delta(summary: pd.DataFrame) -> pd.DataFrame:
    min_n = int(summary["n"].min())
    max_n = int(summary["n"].max())
    rows: list[dict[str, float | str | bool]] = []
    indexed = summary.set_index(["stale_fraction", "strategy", "n"])
    for stale in sorted(summary["stale_fraction"].unique()):
        for strategy in sorted(summary["strategy"].unique()):
            low = indexed.loc[(stale, strategy, min_n)]
            high = indexed.loc[(stale, strategy, max_n)]
            rows.append(
                {
                    "stale_fraction": float(stale),
                    "strategy": strategy,
                    "n_min": min_n,
                    "n_max": max_n,
                    "delta_true_return": float(high["true_return"] - low["true_return"]),
                    "delta_proxy_score": float(high["proxy_score"] - low["proxy_score"]),
                    "delta_proxy_true_gap": float(high["proxy_true_gap"] - low["proxy_true_gap"]),
                    "delta_hallucinated_rollout": float(high["hallucinated_rollout"] - low["hallucinated_rollout"]),
                    "delta_retrieval_precision": float(high["retrieval_precision"] - low["retrieval_precision"])
                    if not pd.isna(high["retrieval_precision"])
                    else np.nan,
                    "proxy_up_true_down": bool(
                        (high["proxy_score"] > low["proxy_score"]) and (high["true_return"] < low["true_return"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def harm_rates(raw: pd.DataFrame) -> pd.DataFrame:
    min_n = int(raw["n"].min())
    max_n = int(raw["n"].max())
    rows: list[dict[str, float | str | int]] = []
    for (stale, strategy), group in raw.groupby(["stale_fraction", "strategy"]):
        low = group[group["n"] == min_n].set_index("trial")
        high = group[group["n"] == max_n].set_index("trial")
        common = sorted(set(low.index).intersection(set(high.index)))
        if not common:
            continue
        harmful = 0
        proxy_gain = 0
        double_fail = 0
        for trial in common:
            true_down = float(high.loc[trial, "true_return"]) < float(low.loc[trial, "true_return"])
            proxy_up = float(high.loc[trial, "proxy_score"]) > float(low.loc[trial, "proxy_score"])
            harmful += int(true_down)
            proxy_gain += int(proxy_up)
            double_fail += int(true_down and proxy_up)
        rows.append(
            {
                "stale_fraction": float(stale),
                "strategy": strategy,
                "trials": len(common),
                "true_harm_rate": harmful / len(common),
                "proxy_gain_rate": proxy_gain / len(common),
                "proxy_gain_true_harm_rate": double_fail / len(common),
            }
        )
    return pd.DataFrame(rows)


def risk_correlations(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    subset = raw[raw["strategy"].isin(["memory_bon", "repaired_memory_bon"])].copy()
    pairs = [
        ("diagnostic_risk", "proxy_true_gap"),
        ("retrieval_stale_rate", "proxy_true_gap"),
        ("retrieval_precision", "proxy_true_gap"),
        ("dropout_variance", "proxy_true_gap"),
        ("retrieval_disagreement", "proxy_true_gap"),
        ("diagnostic_risk", "hallucinated_rollout"),
        ("retrieval_precision", "hallucinated_rollout"),
    ]
    for strategy, group in subset.groupby("strategy"):
        for x_col, y_col in pairs:
            local = group[[x_col, y_col]].dropna()
            if len(local) < 3 or local[x_col].nunique() <= 1 or local[y_col].nunique() <= 1:
                corr = np.nan
            else:
                corr = float(local[x_col].corr(local[y_col]))
            rows.append({"strategy": strategy, "x": x_col, "y": y_col, "pearson_r": corr})
    return pd.DataFrame(rows)


def max_budget_table(summary: pd.DataFrame) -> pd.DataFrame:
    max_n = int(summary["n"].max())
    rows = summary[summary["n"] == max_n].copy()
    rows["label"] = rows["strategy"].map(LABELS).fillna(rows["strategy"])
    return rows[
        [
            "stale_fraction",
            "label",
            "strategy",
            "true_return",
            "proxy_score",
            "proxy_true_gap",
            "hallucinated_rollout",
            "retrieval_precision",
            "retrieval_stale_rate",
            "diagnostic_risk",
        ]
    ]


def write_figures(
    summary: pd.DataFrame,
    ci: pd.DataFrame,
    deltas: pd.DataFrame,
    harms: pd.DataFrame,
    corrs: pd.DataFrame,
    max_table: pd.DataFrame,
) -> list[Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    paths: list[Path] = []

    naive = deltas[deltas["strategy"] == "memory_bon"].sort_values("stale_fraction")
    repaired = deltas[deltas["strategy"] == "repaired_memory_bon"].sort_values("stale_fraction")
    fig, ax1 = plt.subplots(figsize=(6.8, 3.9))
    ax1.axhline(0.0, color="#333333", linewidth=0.8)
    ax1.plot(naive["stale_fraction"], naive["delta_true_return"], marker="o", label="naive true delta")
    ax1.plot(repaired["stale_fraction"], repaired["delta_true_return"], marker="o", label="repair true delta")
    ax1.set_xlabel("stale memory fraction")
    ax1.set_ylabel("true-return change, max N minus N=1")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(naive["stale_fraction"], naive["delta_proxy_score"], marker="s", color="#D95F02", label="naive proxy delta")
    ax2.set_ylabel("naive proxy-score change")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path = RESULTS / "v3_budget_harm_by_staleness.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    main_stale = min(summary["stale_fraction"].unique(), key=lambda value: abs(float(value) - 0.85))
    harm_main = harms[harms["stale_fraction"] == main_stale].copy()
    order = ["base_bon", "memory_bon", "repaired_memory_bon", "oracle_memory_bon"]
    harm_main["order"] = harm_main["strategy"].map({name: i for i, name in enumerate(order)})
    harm_main = harm_main.sort_values("order")
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.bar(
        [LABELS.get(s, s) for s in harm_main["strategy"]],
        harm_main["proxy_gain_true_harm_rate"],
        color=["#4C78A8", "#D95F02", "#1B9E77", "#6A3D9A"],
    )
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("trial rate: proxy improves and true return drops")
    axis.set_title(f"High-staleness amplification rate (stale={main_stale:.2f})")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = RESULTS / "v3_harm_rate_by_strategy.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    corr_plot = corrs[(corrs["y"] == "proxy_true_gap") & (corrs["strategy"].isin(["memory_bon", "repaired_memory_bon"]))].copy()
    corr_plot["name"] = corr_plot["x"].replace(
        {
            "diagnostic_risk": "risk",
            "retrieval_stale_rate": "stale",
            "retrieval_precision": "precision",
            "dropout_variance": "dropout var.",
            "retrieval_disagreement": "disagree",
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    width = 0.36
    names = list(corr_plot["name"].drop_duplicates())
    x = np.arange(len(names))
    for offset, strategy in [(-width / 2, "memory_bon"), (width / 2, "repaired_memory_bon")]:
        vals = [
            float(corr_plot[(corr_plot["strategy"] == strategy) & (corr_plot["name"] == name)]["pearson_r"].iloc[0])
            for name in names
        ]
        axis.bar(x + offset, vals, width=width, label=LABELS[strategy])
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, names)
    axis.set_ylabel("Pearson r with proxy-true gap")
    axis.set_title("Provenance diagnostics predict selected-rollout error")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = RESULTS / "v3_risk_correlation.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    max_n = int(summary["n"].max())
    ci_true = ci[
        (ci["stale_fraction"] == main_stale) & (ci["n"] == max_n) & (ci["metric"] == "true_return")
    ].copy()
    ci_true["order"] = ci_true["strategy"].map({name: i for i, name in enumerate(order)})
    ci_true = ci_true.sort_values("order")
    y = ci_true["mean"].to_numpy(dtype=float)
    low = y - ci_true["ci_low"].to_numpy(dtype=float)
    high = ci_true["ci_high"].to_numpy(dtype=float) - y
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.bar([LABELS.get(s, s) for s in ci_true["strategy"]], y, color=["#4C78A8", "#D95F02", "#1B9E77", "#6A3D9A"])
    axis.errorbar(np.arange(len(y)), y, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=3)
    axis.set_ylabel("true return at max N with bootstrap 95% CI")
    axis.set_title("Main stale-memory condition")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = RESULTS / "v3_bootstrap_true_return.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    max_raw = max_table[max_table["stale_fraction"] == main_stale].copy()
    fig, axis = plt.subplots(figsize=(6.9, 3.9))
    for strategy, color in [("memory_bon", "#D95F02"), ("repaired_memory_bon", "#1B9E77"), ("oracle_memory_bon", "#6A3D9A")]:
        group = max_table[max_table["strategy"] == strategy].sort_values("stale_fraction")
        axis.plot(group["retrieval_precision"], group["proxy_true_gap"], marker="o", label=LABELS[strategy], color=color)
    axis.set_xlabel("selected retrieval precision at max N")
    axis.set_ylabel("proxy-true gap at max N")
    axis.set_title("Precision gap across staleness regimes")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = RESULTS / "v3_precision_gap_tradeoff.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    for path in paths:
        shutil.copy2(path, FIGURES / path.name)
    return paths


def write_tables(
    ci: pd.DataFrame,
    deltas: pd.DataFrame,
    harms: pd.DataFrame,
    corrs: pd.DataFrame,
    max_table: pd.DataFrame,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ci.to_csv(RESULTS / "bootstrap_ci.csv", index=False)
    deltas.to_csv(RESULTS / "budget_delta.csv", index=False)
    harms.to_csv(RESULTS / "harm_rates.csv", index=False)
    corrs.to_csv(RESULTS / "risk_correlations.csv", index=False)
    max_table.to_csv(RESULTS / "max_budget_strategy_table.csv", index=False)


def write_macros(summary: pd.DataFrame, ci: pd.DataFrame, deltas: pd.DataFrame, harms: pd.DataFrame, corrs: pd.DataFrame) -> dict:
    main_stale = min(summary["stale_fraction"].unique(), key=lambda value: abs(float(value) - 0.85))
    max_n = int(summary["n"].max())
    min_n = int(summary["n"].min())
    main = summary[(summary["stale_fraction"] == main_stale) & (summary["n"] == max_n)].set_index("strategy")
    main_low = summary[(summary["stale_fraction"] == main_stale) & (summary["n"] == min_n)].set_index("strategy")
    delta_main = deltas[deltas["stale_fraction"] == main_stale].set_index("strategy")
    harm_main = harms[harms["stale_fraction"] == main_stale].set_index("strategy")

    ci_true = ci[
        (ci["stale_fraction"] == main_stale)
        & (ci["n"] == max_n)
        & (ci["metric"] == "true_return")
    ].set_index("strategy")
    ci_gap = ci[
        (ci["stale_fraction"] == main_stale)
        & (ci["n"] == max_n)
        & (ci["metric"] == "proxy_true_gap")
    ].set_index("strategy")
    corr_risk_gap = corrs[
        (corrs["strategy"] == "memory_bon") & (corrs["x"] == "diagnostic_risk") & (corrs["y"] == "proxy_true_gap")
    ]["pearson_r"].iloc[0]
    corr_precision_gap = corrs[
        (corrs["strategy"] == "memory_bon") & (corrs["x"] == "retrieval_precision") & (corrs["y"] == "proxy_true_gap")
    ]["pearson_r"].iloc[0]

    stale_levels = int(summary["stale_fraction"].nunique())
    raw_rows = int(len(pd.read_csv(ROOT / "results" / "v3_base" / "raw_rollouts.csv")))
    harmful_levels = int(deltas[(deltas["strategy"] == "memory_bon") & (deltas["proxy_up_true_down"])].shape[0])

    summary_json = {
        "raw_rows": raw_rows,
        "summary_rows": int(len(summary)),
        "stale_levels": stale_levels,
        "main_stale": float(main_stale),
        "min_n": min_n,
        "max_n": max_n,
        "naive": {
            "n1_true": float(main_low.loc["memory_bon", "true_return"]),
            "max_true": float(main.loc["memory_bon", "true_return"]),
            "max_true_ci_low": float(ci_true.loc["memory_bon", "ci_low"]),
            "max_true_ci_high": float(ci_true.loc["memory_bon", "ci_high"]),
            "max_gap": float(main.loc["memory_bon", "proxy_true_gap"]),
            "max_gap_ci_low": float(ci_gap.loc["memory_bon", "ci_low"]),
            "max_gap_ci_high": float(ci_gap.loc["memory_bon", "ci_high"]),
            "delta_true": float(delta_main.loc["memory_bon", "delta_true_return"]),
            "delta_proxy": float(delta_main.loc["memory_bon", "delta_proxy_score"]),
            "harm_rate": float(harm_main.loc["memory_bon", "proxy_gain_true_harm_rate"]),
            "retrieval_precision": float(main.loc["memory_bon", "retrieval_precision"]),
            "stale_rate": float(main.loc["memory_bon", "retrieval_stale_rate"]),
        },
        "repaired": {
            "max_true": float(main.loc["repaired_memory_bon", "true_return"]),
            "max_true_ci_low": float(ci_true.loc["repaired_memory_bon", "ci_low"]),
            "max_true_ci_high": float(ci_true.loc["repaired_memory_bon", "ci_high"]),
            "gain_over_naive": float(main.loc["repaired_memory_bon", "true_return"] - main.loc["memory_bon", "true_return"]),
            "harm_rate": float(harm_main.loc["repaired_memory_bon", "proxy_gain_true_harm_rate"]),
            "retrieval_precision": float(main.loc["repaired_memory_bon", "retrieval_precision"]),
            "stale_rate": float(main.loc["repaired_memory_bon", "retrieval_stale_rate"]),
            "hallucination": float(main.loc["repaired_memory_bon", "hallucinated_rollout"]),
        },
        "oracle": {
            "max_true": float(main.loc["oracle_memory_bon", "true_return"]),
            "gap_after_repair": float(main.loc["oracle_memory_bon", "true_return"] - main.loc["repaired_memory_bon", "true_return"]),
        },
        "diagnostics": {
            "risk_gap_corr": float(corr_risk_gap),
            "precision_gap_corr": float(corr_precision_gap),
            "harmful_stale_levels": harmful_levels,
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8")

    text = ""
    text += _macro("VThreeRawRows", f"{raw_rows:,}")
    text += _macro("VThreeSummaryRows", f"{len(summary):,}")
    text += _macro("VThreeStaleLevels", str(stale_levels))
    text += _macro("VThreeMainStale", _fmt(main_stale, 2))
    text += _macro("VThreeMinN", str(min_n))
    text += _macro("VThreeMaxN", str(max_n))
    text += _macro("VThreeNaiveNOneTrue", _fmt(main_low.loc["memory_bon", "true_return"], 2))
    text += _macro("VThreeNaiveMaxTrue", _fmt(main.loc["memory_bon", "true_return"], 2))
    text += _macro("VThreeNaiveMaxTrueCILow", _fmt(ci_true.loc["memory_bon", "ci_low"], 2))
    text += _macro("VThreeNaiveMaxTrueCIHigh", _fmt(ci_true.loc["memory_bon", "ci_high"], 2))
    text += _macro("VThreeNaiveDeltaTrue", _fmt(delta_main.loc["memory_bon", "delta_true_return"], 2))
    text += _macro("VThreeNaiveDeltaProxy", _fmt(delta_main.loc["memory_bon", "delta_proxy_score"], 2))
    text += _macro("VThreeNaiveMaxGap", _fmt(main.loc["memory_bon", "proxy_true_gap"], 2))
    text += _macro("VThreeNaiveMaxGapCILow", _fmt(ci_gap.loc["memory_bon", "ci_low"], 2))
    text += _macro("VThreeNaiveMaxGapCIHigh", _fmt(ci_gap.loc["memory_bon", "ci_high"], 2))
    text += _macro("VThreeNaiveHarmRate", _pct(harm_main.loc["memory_bon", "proxy_gain_true_harm_rate"]))
    text += _macro("VThreeNaivePrecision", _pct(main.loc["memory_bon", "retrieval_precision"]))
    text += _macro("VThreeNaiveStaleRate", _pct(main.loc["memory_bon", "retrieval_stale_rate"]))
    text += _macro("VThreeNaiveHallucination", _pct(main.loc["memory_bon", "hallucinated_rollout"]))
    text += _macro("VThreeBaseMaxTrue", _fmt(main.loc["base_bon", "true_return"], 2))
    text += _macro("VThreeRepairedMaxTrue", _fmt(main.loc["repaired_memory_bon", "true_return"], 2))
    text += _macro("VThreeRepairedMaxTrueCILow", _fmt(ci_true.loc["repaired_memory_bon", "ci_low"], 2))
    text += _macro("VThreeRepairedMaxTrueCIHigh", _fmt(ci_true.loc["repaired_memory_bon", "ci_high"], 2))
    text += _macro("VThreeRepairGain", _fmt(main.loc["repaired_memory_bon", "true_return"] - main.loc["memory_bon", "true_return"], 2))
    text += _macro("VThreeRepairedHarmRate", _pct(harm_main.loc["repaired_memory_bon", "proxy_gain_true_harm_rate"]))
    text += _macro("VThreeRepairedPrecision", _pct(main.loc["repaired_memory_bon", "retrieval_precision"]))
    text += _macro("VThreeRepairedStaleRate", _pct(main.loc["repaired_memory_bon", "retrieval_stale_rate"]))
    text += _macro("VThreeRepairedHallucination", _pct(main.loc["repaired_memory_bon", "hallucinated_rollout"]))
    text += _macro("VThreeOracleMaxTrue", _fmt(main.loc["oracle_memory_bon", "true_return"], 2))
    text += _macro("VThreeOracleGap", _fmt(main.loc["oracle_memory_bon", "true_return"] - main.loc["repaired_memory_bon", "true_return"], 2))
    text += _macro("VThreeRiskGapCorr", _fmt(corr_risk_gap, 2))
    text += _macro("VThreePrecisionGapCorr", _fmt(corr_precision_gap, 2))
    text += _macro("VThreeHarmfulStaleLevels", f"{harmful_levels}/{stale_levels}")
    (PAPER / "v3_results_macros.tex").write_text(text, encoding="utf-8")
    return summary_json


def main() -> None:
    raw, summary = load_inputs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ci = bootstrap_ci(raw)
    deltas = budget_delta(summary)
    harms = harm_rates(raw)
    corrs = risk_correlations(raw)
    max_table = max_budget_table(summary)
    write_tables(ci, deltas, harms, corrs, max_table)
    figures = write_figures(summary, ci, deltas, harms, corrs, max_table)
    summary_json = write_macros(summary, ci, deltas, harms, corrs)
    print(f"v3 cached evidence complete: {RESULTS}")
    print(f"raw_rows={summary_json['raw_rows']} figures={len(figures)}")


if __name__ == "__main__":
    main()
