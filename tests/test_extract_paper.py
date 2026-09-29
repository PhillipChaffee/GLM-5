"""Checks for the paper-text sidecar writer."""

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter

from scripts.extract_paper import extract_paper


def test_extract_paper_removes_nuls_and_keeps_page_boundaries(tmp_path: Path) -> None:
    """A page with no extractable text still keeps its separator."""
    src = tmp_path / "paper.pdf"
    first = MagicMock()
    first.extract_text.return_value = "first\x00 page"
    empty = MagicMock()
    empty.extract_text.return_value = None
    last = MagicMock()
    last.extract_text.return_value = "last page"
    with patch("scripts.extract_paper.PdfReader") as reader:
        reader.return_value.pages = [first, empty, last]
        out = extract_paper(src)
    reader.assert_called_once_with(src)
    assert out == src.with_suffix(".txt")
    assert out.read_text() == "first page\n\nlast page"


def test_module_entry_point_writes_sidecar(tmp_path: Path) -> None:
    """The shell wrapper's module invocation writes a sidecar."""
    src = tmp_path / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with src.open("wb") as stream:
        writer.write(stream)
    with patch.object(sys, "argv", ["scripts.extract_paper", str(src)]):
        runpy.run_path("scripts/extract_paper.py", run_name="__main__")
    assert src.with_suffix(".txt").read_text() == ""
