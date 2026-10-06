"""Extract searchable, NUL-free text from a paper PDF."""

import sys
from pathlib import Path

from pypdf import PdfReader


def extract_paper(src: Path) -> Path:
    """Write a NUL-free, hook-clean text sidecar beside a PDF and return its path."""
    text = "\n".join((page.extract_text() or "") for page in PdfReader(src).pages)
    text = text.replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    out = src.with_suffix(".txt")
    out.write_text(text.rstrip() + "\n")
    sys.stdout.write(f"{src} -> {out} ({len(text)} chars)\n")
    return out


if __name__ == "__main__":
    extract_paper(Path(sys.argv[1]))
