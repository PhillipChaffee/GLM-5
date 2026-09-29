"""Extract searchable, NUL-free text from a paper PDF."""

import sys
from pathlib import Path

from pypdf import PdfReader


def extract_paper(src: Path) -> Path:
    """Write extracted text beside a PDF and return the sidecar path."""
    text = "\n".join((page.extract_text() or "") for page in PdfReader(src).pages)
    text = text.replace("\x00", "")
    out = src.with_suffix(".txt")
    out.write_text(text)
    sys.stdout.write(f"{src} -> {out} ({len(text)} chars)\n")
    return out


if __name__ == "__main__":
    extract_paper(Path(sys.argv[1]))
