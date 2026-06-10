from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_iclr_anonymous_style_files_present():
    assert (ROOT / "paper" / "iclr2026_conference.sty").exists()
    assert (ROOT / "paper" / "iclr2026_conference.bst").exists()


def test_main_tex_keeps_finalcopy_commented_when_present():
    main = ROOT / "paper" / "main.tex"
    if main.exists():
        text = main.read_text(encoding="utf-8")
        assert "\\usepackage{iclr2026_conference,times}" in text
        assert "%\\iclrfinalcopy" in text
