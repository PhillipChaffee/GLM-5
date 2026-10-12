"""Pytest wiring for the GLM-5.3-Flash drafter experiments.

The experiment modules import the pinned DeepSpec clone with the worktree
overlay copied over it, so the tests assemble that combined package in a
scratch directory and put it first on ``sys.path``. Set ``DEEPSPEC_CLONE`` to
the pinned clone location (default is this machine's scratch path) and
``GLM_TOKENIZER_DIR`` to the directory holding ``tokenizer.json`` plus
``chat_template.jinja``. Collection is skipped when either asset is absent.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

TESTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TESTS_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
OVERLAY_DIR = EXPERIMENT_DIR / "deepspec_glm"
CLONE_DIR = Path(
    os.environ.get(
        "DEEPSPEC_CLONE",
        "/tmp/opencode-scratch/GLM-5/opencode-91382/DeepSpec",
    )
)
COMBINED_DIR = Path(
    os.environ.get(
        "GLM_DEEPSPEC_COMBINED",
        "/tmp/opencode-scratch/GLM-5/deepspec-combined",
    )
)
TOKENIZER_DIR = Path(
    os.environ.get(
        "GLM_TOKENIZER_DIR",
        "/tmp/opencode-scratch/GLM-5/opencode-91382",
    )
)


def _require(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed in this environment", allow_module_level=True)


_require("torch")
_require("transformers")


def _build_combined_package() -> str:
    clone_pkg = CLONE_DIR / "deepspec"
    overlay_pkg = OVERLAY_DIR / "deepspec"
    if not clone_pkg.is_dir():
        pytest.skip(
            f"DeepSpec clone not found at {CLONE_DIR} (set DEEPSPEC_CLONE)",
            allow_module_level=True,
        )
    combined_pkg = COMBINED_DIR / "deepspec"
    if combined_pkg.exists():
        shutil.rmtree(combined_pkg)
    combined_pkg.mkdir(parents=True)
    shutil.copytree(clone_pkg, combined_pkg, dirs_exist_ok=True)
    shutil.copytree(overlay_pkg, combined_pkg, dirs_exist_ok=True)
    if str(COMBINED_DIR) in sys.path:
        sys.path.remove(str(COMBINED_DIR))
    sys.path.insert(0, str(COMBINED_DIR))
    return str(COMBINED_DIR)


_build_combined_package()


@pytest.fixture(scope="session")
def glm_tokenizer() -> "PreTrainedTokenizerFast":
    """Load the real GLM-5.3-Flash tokenizer with the spike chat template."""
    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    chat_template = TOKENIZER_DIR / "chat_template.jinja"
    if not tokenizer_json.is_file() or not chat_template.is_file():
        pytest.skip(
            f"GLM tokenizer assets not found under {TOKENIZER_DIR} (set GLM_TOKENIZER_DIR)",
        )
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
    tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    return tokenizer
