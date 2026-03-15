"""Pytest fixtures for scenario-driven accuracy tests.

These fixtures discover converted output artefacts that were produced by the
``convert.py`` pipeline.  If the ``output/`` directory does not exist the
accuracy tests are automatically skipped — there is nothing to test without
a prior conversion run.

Expected output layout produced by the pipeline:

    output/
        fortran/blas/          ← original .f source files
        rust/blas_converted/   ← the Rust crate with converted .rs sources
        datasets/              ← shared dataset files (created at test time)
        fortran_drivers/       ← compiled Fortran test drivers (created at test time)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"
_FORTRAN_SRC_DIR = _OUTPUT_DIR / "fortran" / "blas"
_RUST_CRATE_DIR = _OUTPUT_DIR / "rust" / "blas_converted"
_DATASETS_DIR = _OUTPUT_DIR / "datasets"
_FORTRAN_REF_DIR = _OUTPUT_DIR / "fortran_drivers"


# ---------------------------------------------------------------------------
# Session-scoped output-directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Return the pipeline output directory, skipping if it does not exist."""
    if not _OUTPUT_DIR.exists():
        pytest.skip(
            "Pipeline output directory 'output/' not found. "
            "Run `python convert.py` first to generate converted sources."
        )
    return _OUTPUT_DIR


@pytest.fixture(scope="session")
def fortran_src_dir(output_dir: Path) -> Path:  # noqa: ARG001
    """Return the directory containing the original Fortran .f sources."""
    if not _FORTRAN_SRC_DIR.exists():
        pytest.skip(f"Fortran source directory not found: {_FORTRAN_SRC_DIR}")
    return _FORTRAN_SRC_DIR


@pytest.fixture(scope="session")
def rust_crate_dir(output_dir: Path) -> Path:  # noqa: ARG001
    """Return the Rust crate directory containing converted sources."""
    if not _RUST_CRATE_DIR.exists():
        pytest.skip(f"Rust crate directory not found: {_RUST_CRATE_DIR}")
    return _RUST_CRATE_DIR


# ---------------------------------------------------------------------------
# Converted-function discovery
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def converted_function_names(
    fortran_src_dir: Path,
    rust_crate_dir: Path,
) -> List[str]:
    """Return names of functions that have both a Fortran source and a Rust crate.

    A function qualifies when:
    - A ``.f`` file exists under *fortran_src_dir*.
    - The Rust crate directory exists (a single crate may contain all functions).

    The list is sorted for deterministic parametrisation.
    """
    rust_src_dir = rust_crate_dir / "src"
    fortran_stems = {p.stem.lower() for p in _FORTRAN_SRC_DIR.glob("*.f")}
    # Rust sources: each function gets its own .rs file under src/
    rust_stems = {p.stem.lower() for p in rust_src_dir.glob("*.rs")
                  if p.stem not in ("lib", "main")} if rust_src_dir.exists() else set()
    # Functions present in both
    common = sorted(fortran_stems & rust_stems)
    if not common:
        pytest.skip(
            "No functions found in both Fortran and Rust sources. "
            "Run `python convert.py` first."
        )
    return common


# ---------------------------------------------------------------------------
# Per-function fixtures returned as factory helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fortran_source_for(fortran_src_dir: Path):
    """Return a callable ``(fn_name) -> Path`` for Fortran source look-up."""
    def _lookup(fn_name: str) -> Optional[Path]:
        p = fortran_src_dir / f"{fn_name.lower()}.f"
        return p if p.exists() else None
    return _lookup


@pytest.fixture(scope="session")
def datasets_dir(output_dir: Path) -> Path:  # noqa: ARG001
    """Return (and create) the directory used for dataset files."""
    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return _DATASETS_DIR


@pytest.fixture(scope="session")
def fortran_ref_dir(output_dir: Path) -> Path:  # noqa: ARG001
    """Return (and create) the directory for compiled Fortran test drivers."""
    _FORTRAN_REF_DIR.mkdir(parents=True, exist_ok=True)
    return _FORTRAN_REF_DIR
