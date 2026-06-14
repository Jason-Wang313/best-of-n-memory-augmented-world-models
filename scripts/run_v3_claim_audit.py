from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
REPO_PDF = ROOT / "paper" / "final" / "best-of-n-memory-augmented-world-models-v3.pdf"
DESKTOP_PDF = DESKTOP / "best-of-n-memory-augmented-world-models-v3.pdf"
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SUMMARY = ROOT / "results" / "v3_cached_evidence" / "summary.json"
LATEX_LOG = ROOT / "paper" / "main.log"
OLD_GENERIC_PDF = ROOT / "paper" / ("best-of-n-memory-augmented-world-models" + ".pdf")

STALE_PATTERNS = (
    "best-of-n-memory-augmented-world-models-" + "v" + "2.pdf",
    "best-of-n-memory-augmented-world-models" + ".pdf",
    "v" + "2 artifact",
    "v" + "2 review",
)

SCAN_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "paper",
    ROOT / "scripts",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: Path) -> int:
    try:
        completed = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raw = path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page\b", raw))
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
    if not match:
        fail(f"could not parse page count for {path}")
    return int(match.group(1))


def iter_text_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".tex", ".json", ".py", ".csv"}:
            yield path


def audit_pdfs() -> None:
    if OLD_GENERIC_PDF.exists():
        fail(f"old generic paper PDF still exists: {OLD_GENERIC_PDF}")
    if not REPO_PDF.exists():
        fail(f"missing repo final PDF: {REPO_PDF}")
    if not DESKTOP_PDF.exists():
        fail(f"missing Desktop PDF: {DESKTOP_PDF}")
    repo_pages = page_count(REPO_PDF)
    desktop_pages = page_count(DESKTOP_PDF)
    if repo_pages < 25:
        fail(f"repo PDF has {repo_pages} pages, expected at least 25")
    if repo_pages != desktop_pages:
        fail(f"repo/Desktop page counts differ: {repo_pages} vs {desktop_pages}")
    if sha256(REPO_PDF) != sha256(DESKTOP_PDF):
        fail("repo and Desktop PDF hashes differ")


def audit_evidence() -> None:
    if not SUMMARY.exists():
        fail(f"missing v3 evidence summary: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if int(summary.get("raw_rows", 0)) != 1152:
        fail("unexpected raw rollout count")
    if int(summary.get("max_n", 0)) != 64:
        fail("unexpected maximum candidate budget")
    if abs(float(summary.get("main_stale", 0.0)) - 0.85) > 1e-9:
        fail("unexpected main stale fraction")
    naive = summary.get("naive", {})
    repaired = summary.get("repaired", {})
    diagnostics = summary.get("diagnostics", {})
    if float(naive.get("delta_true", 0.0)) >= -1.0:
        fail("naive memory does not show true-return budget harm")
    if float(naive.get("delta_proxy", 0.0)) <= 1.0:
        fail("naive memory does not show proxy gain under larger budget")
    if float(naive.get("harm_rate", 0.0)) < 0.5:
        fail("naive proxy-up/true-down harm rate is too low")
    if float(repaired.get("gain_over_naive", 0.0)) < 8.0:
        fail("repair gain over naive memory is below v3 expectation")
    if int(diagnostics.get("harmful_stale_levels", 0)) < 3:
        fail("budget harm does not appear across enough stale settings")


def audit_stale_text() -> None:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        for path in iter_text_files(root):
            if path == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in STALE_PATTERNS:
                if pattern in text:
                    hits.append(f"{path.relative_to(ROOT)}: {pattern}")
    if hits:
        fail("stale v2/current-artifact text remains:\n" + "\n".join(hits))


def audit_latex_log() -> None:
    if not LATEX_LOG.exists():
        fail(f"missing LaTeX log: {LATEX_LOG}")
    text = LATEX_LOG.read_text(encoding="utf-8", errors="ignore")
    bad = re.findall(
        r"(undefined|Citation|Overfull|Fatal|Emergency|LaTeX Warning|Package natbib Warning|Package hyperref Warning)",
        text,
        flags=re.IGNORECASE,
    )
    if bad:
        fail(f"LaTeX log contains blocking warnings: {sorted(set(bad))}")


def audit_source_map() -> None:
    if not SOURCE_MAP.exists():
        fail(f"missing source map: {SOURCE_MAP}")
    text = SOURCE_MAP.read_text(encoding="utf-8")
    expected = (
        "best-of-n-memory-augmented-world-models-v3.pdf",
        "C:\\Users\\wangz\\best-of-n-memory-augmented-world-models",
        "Jason-Wang313/best-of-n-memory-augmented-world-models",
    )
    if not all(part in text for part in expected):
        fail("source map does not point the v3 Desktop PDF to this folder and repo")


def main() -> None:
    audit_pdfs()
    audit_evidence()
    audit_stale_text()
    audit_latex_log()
    audit_source_map()
    print("submission audit complete: memory-impostor v3")


if __name__ == "__main__":
    main()
