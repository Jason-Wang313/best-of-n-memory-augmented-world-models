"""Build the anonymous ICLR paper and copy the versioned PDF to the Desktop."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DOCS = ROOT / "docs"
ONEDRIVE_DESKTOP = Path.home() / "OneDrive" / "Desktop"
DESKTOP = ONEDRIVE_DESKTOP if ONEDRIVE_DESKTOP.exists() else Path.home() / "Desktop"
DESKTOP_PDF = DESKTOP / "best-of-n-memory-augmented-world-models-v4.pdf"
FINAL_PDF = PAPER / "final" / "best-of-n-memory-augmented-world-models-v4.pdf"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def _summary_path() -> Path:
    for rel in ("results/v4_base/summary.csv", "results/paper/summary.csv", "results/full/summary.csv", "results/smoke/summary.csv"):
        path = ROOT / rel
        if path.exists():
            return path
    raise FileNotFoundError("missing experiment summary; expected results/v4_base/summary.csv")


def write_result_macros() -> Path:
    path = _summary_path()
    summary = pd.read_csv(path)
    stale = min(summary["stale_fraction"].unique(), key=lambda value: abs(float(value) - 0.85))
    main = summary[summary["stale_fraction"] == stale]
    max_n = int(main["n"].max())
    at_max = main[main["n"] == max_n].set_index("strategy")
    at_one = main[main["n"] == 1].set_index("strategy")
    naive_one = at_one.loc["memory_bon"]
    naive = at_max.loc["memory_bon"]
    base = at_max.loc["base_bon"]
    repaired = at_max.loc["repaired_memory_bon"]
    oracle = at_max.loc["oracle_memory_bon"]

    text = ""
    text += macro("ResultPreset", path.parent.name)
    text += macro("MainStaleFraction", f"{float(stale):.2f}")
    text += macro("MaxN", str(max_n))
    text += macro("NaiveNOneTrue", f"{float(naive_one['true_return']):.2f}")
    text += macro("NaiveMaxTrue", f"{float(naive['true_return']):.2f}")
    text += macro("NaiveMaxProxy", f"{float(naive['proxy_score']):.2f}")
    text += macro("NaiveMaxGap", f"{float(naive['proxy_true_gap']):.2f}")
    text += macro("NaiveMaxHallucination", f"{100.0 * float(naive['hallucinated_rollout']):.1f}\\%")
    text += macro("NaiveMaxPrecision", f"{100.0 * float(naive['retrieval_precision']):.1f}\\%")
    text += macro("NaiveMaxStale", f"{100.0 * float(naive['retrieval_stale_rate']):.1f}\\%")
    text += macro("BaseMaxTrue", f"{float(base['true_return']):.2f}")
    text += macro("RepairedMaxTrue", f"{float(repaired['true_return']):.2f}")
    text += macro("RepairedMaxHallucination", f"{100.0 * float(repaired['hallucinated_rollout']):.1f}\\%")
    text += macro("RepairedMaxPrecision", f"{100.0 * float(repaired['retrieval_precision']):.1f}\\%")
    text += macro("RepairedMaxStale", f"{100.0 * float(repaired['retrieval_stale_rate']):.1f}\\%")
    text += macro("OracleMaxTrue", f"{float(oracle['true_return']):.2f}")
    text += macro("RepairGain", f"{float(repaired['true_return'] - naive['true_return']):.2f}")
    text += macro("OracleGap", f"{float(oracle['true_return'] - repaired['true_return']):.2f}")
    (PAPER / "results_macros.tex").write_text(text, encoding="utf-8")
    return path


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PAPER, check=True, text=True, capture_output=True)


def write_v4_protocol_evidence() -> None:
    subprocess.run(
        ["python", str(ROOT / "experiments" / "18_v4_protocol_evidence.py")],
        cwd=ROOT,
        check=True,
        text=True,
    )


def write_v4_standard_benchmarks() -> None:
    for script in ("20_v4_toytext_memory_benchmarks.py", "19_v4_gymnasium_memory_benchmarks.py"):
        subprocess.run(
            ["python", str(ROOT / "experiments" / script)],
            cwd=ROOT,
            check=True,
            text=True,
        )


def run_latex() -> Path:
    errors: list[str] = []
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out", ".pdf"):
        artifact = PAPER / f"main{suffix}"
        if artifact.exists():
            artifact.unlink()

    latexmk = shutil.which("latexmk")
    if latexmk is not None:
        try:
            _run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        except subprocess.CalledProcessError as exc:
            errors.append(f"latexmk failed\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}")

    if not (PAPER / "main.pdf").exists():
        try:
            _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
            _run(["bibtex", "main"])
            _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
            _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
            _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        except subprocess.CalledProcessError as exc:
            errors.append(f"pdflatex/bibtex failed\nCOMMAND: {' '.join(exc.cmd)}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}")

    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        DOCS.mkdir(exist_ok=True)
        failure = DOCS / "paper_build_failure.md"
        failure.write_text("# Paper Build Failure\n\n" + "\n\n".join(errors), encoding="utf-8")
        raise RuntimeError(f"paper build failed; see {failure}")

    DESKTOP_PDF.parent.mkdir(exist_ok=True)
    FINAL_PDF.parent.mkdir(exist_ok=True)
    shutil.copy2(pdf, DESKTOP_PDF)
    shutil.copy2(pdf, FINAL_PDF)
    return DESKTOP_PDF


def main() -> None:
    write_v4_protocol_evidence()
    write_v4_standard_benchmarks()
    summary = write_result_macros()
    pdf = run_latex()
    print(f"used {summary}")
    print(f"wrote {pdf}")
    print(f"wrote {FINAL_PDF}")


if __name__ == "__main__":
    main()
