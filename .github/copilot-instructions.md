# Copilot Instructions

## Commands

```bash
# Install in editable mode
pip install -e .

# Interactive wizard (full pipeline)
python convert.py

# Non-interactive / CI smoke test (converts dgemm with Strategy 3)
python convert.py --non-interactive

# Convert specific functions
python convert.py --non-interactive --functions dgemm,ddot --strategy 1

# Convert a random sample of N functions
python convert.py --non-interactive --functions 10
```

No dedicated test suite yet. `python convert.py --non-interactive` is the primary integration test.

## Architecture

The pipeline flows in a single direction:

```
fetch Fortran source → parse → convert (strategy) → scaffold Cargo crate → accuracy check → benchmark → report
```

**Entry points**
- `convert.py` — CLI entry point; handles arg parsing and non-interactive mode
- `fortran_to_rust/cli.py` — Interactive wizard using Rich prompts

**Core modules**
- `fetcher.py` — Downloads BLAS `.f` source files from netlib/LAPACK GitHub; always fetches `lsame` + `xerbla` as link-time support routines alongside any user-requested functions
- `parser.py` — Regex-based Fortran-77 fixed-form parser; produces `FortranRoutine` objects
- `strategies/` — Three conversion strategies all sharing `ConversionStrategy` base class:
  - `1` / `LLMFirstStrategy` — LLM translates, then 2-round compile-repair loop; falls back to Hybrid if LLM is unavailable
  - `2` / `AgenticStrategy` — Questioner–Solver LLM pair, up to 3 clarifying rounds, then idiomisation pass
  - `3` / `HybridStrategy` — f2c → rule-based C→Rust → optional LLM polish (works offline)
- `rust_project.py` — Scaffolds the Cargo crate; runs `cargo check/build/test/clippy`
- `test_harness.py` — Generates and runs Fortran + Rust drivers for numerical accuracy comparison
- `benchmarker.py` — Wall-clock comparison: `gfortran -O2` vs `cargo --release`
- `reporter.py` — Writes `output/reports/<timestamp>_report.md` + `.html`; returns `Tuple[Path, Path]`
- `llm_client.py` — Thin HTTP wrapper for GitHub Models / OpenAI / compatible endpoints

## Key Conventions

### LLM output normalization
After every LLM call (translation or repair), `_ensure_top_level_pub_fn(source, fn_lower)` is applied to ensure a bare `fn {name}` at line-start is promoted to `pub fn {name}`. LLM prompts always require the function to be a **top-level `pub fn` (or `pub unsafe fn`) — never wrapped inside a `mod` block**.

### Fortran type mapping
| Fortran | Rust |
|---------|------|
| `DOUBLE PRECISION` | `f64` |
| `REAL` | `f32` |
| `INTEGER` | `i32` |
| `LOGICAL` | `bool` |
| `CHARACTER*1` | `u8` (pass ASCII byte literals like `b'N'`) |

This mapping must be consistent across LLM prompts (`llm_client.py`), the hybrid stub generator (`strategies/hybrid.py`), and the accuracy example generator (`test_harness.py`).

### Cargo crate layout
`scaffold_crate()` writes each function to `src/{fn_lower}.rs` and generates `src/lib.rs` with both:
```rust
pub mod dgemm;
pub use dgemm::*;  // flat re-export so accuracy examples can `use blas_converted::*`
```
The lib header always includes `#![allow(clippy::all, non_snake_case, unused_variables, dead_code)]`.

### Accuracy test harness
- Fortran and Rust drivers use `random.Random(test_index)` for deterministic seeding, iterating args in declaration order
- Real scalars: `uniform(0.5, 2.0)`; real array elements: `uniform(-1.0, 1.0)`; integers/chars/logicals use fixed values
- Accuracy example Rust code wraps the function call in `unsafe { }` to handle both safe and LLM-generated unsafe functions without `E0133`
- Accuracy tolerance: `1e-10` absolute error (`_TOLERANCE` in `test_harness.py`)

### Support routines
`lsame` and `xerbla` are always fetched alongside user-selected functions for gfortran linking, but they are **excluded from `functions_to_convert`** — they are never passed through the conversion strategies.

### LLM authentication
Token resolution order for `github_models` provider:
1. `LLM_API_KEY` env var / `.env` file
2. `gh auth token` (GitHub CLI)
3. `~/.config/github-copilot/hosts.json`

`copilot` is a backward-compatible alias for `github_models` — same endpoint, same token format. `.env` is loaded automatically without `python-dotenv`; never commit it.

### Progress callbacks
All strategy `convert()` calls accept an optional `progress_callback: callable(message: str)`. Pass one to stream status messages to the CLI or capture them for reports.
