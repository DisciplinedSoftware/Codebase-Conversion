# Codebase-Conversion

Automated pipeline for converting Fortran codebases to idiomatic Rust, with
post-conversion accuracy and performance reports.

The first supported library is **BLAS** (Basic Linear Algebra Subprograms).
The primary milestone is full automation of `dgemm`; the pipeline then scales
to 10 functions and eventually the full library.

---

## Quick start

### Option A — GitHub Codespaces / VS Code devcontainer (recommended)

Click **"Open in Codespaces"** (or **"Reopen in Container"** in VS Code).
All dependencies — Python 3.12, Rust, `gfortran`, `f2c` — are pre-installed
and the package is automatically installed in editable mode.

```
# Nothing to install — just run:
python convert.py
```

### Option B — Local setup

**Prerequisites**

| Tool | Minimum version |
|------|----------------|
| Python | 3.10 |
| Rust / Cargo | 1.70 |
| gfortran | 9 |
| f2c | any |

```bash
# 1. Clone
git clone https://github.com/DisciplinedSoftware/Codebase-Conversion
cd Codebase-Conversion

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. (Optional) install as editable package
pip install -e .

# 4. Run the interactive wizard
python convert.py
```

---

## Usage

### Interactive wizard

```
python convert.py
```

The wizard guides you through:

1. **Library** — currently BLAS
2. **Scope** — single function · 10-function sample · full library
3. **Strategy** — choose one of the three conversion approaches below
4. **LLM configuration** — detected automatically from the environment
5. **Live conversion** — progress feedback, inline Q&A (Strategy 2), repair rounds
6. **Cargo crate** — the output is packaged as a ready-to-use crate
7. **Accuracy check** — numerical comparison of Fortran vs Rust outputs
8. **Performance benchmark** — wall-clock comparison (`gfortran -O2` vs `cargo --release`)
9. **Markdown report** — written to `output/reports/<timestamp>_report.md`

### Non-interactive / CI mode

```bash
python convert.py --non-interactive          # converts dgemm with the Hybrid strategy
python convert.py --output-dir /tmp/output   # custom output directory
```

---

## Conversion strategies

### Strategy 1 — LLM-First with Rule Fallback *(requires LLM key)*

1. Pre-process — strip comments, build call graph
2. LLM translates each function (≤ 200-line chunks)
3. `cargo check` → feed errors back to LLM for up to 2 repair rounds
4. Hybrid rule-based fallback if LLM is unavailable

### Strategy 2 — Agentic Multi-Turn Dialogue *(requires LLM key)*

1. **Solver** LLM proposes an initial Rust translation
2. **Questioner** LLM asks up to 3 clarifying questions about Fortran intrinsics,
   implicit typing, and module interfaces; the Solver revises
3. Compile-repair loop (same as Strategy 1)
4. Idiomisation pass — LLM replaces `unsafe` blocks with safe Rust idioms

### Strategy 3 — Hybrid Rule-Based + LLM Polish *(works fully offline)*

1. **f2c** converts Fortran → C (deterministic skeleton)
2. Rule-based C → Rust transformation
3. Optional **LLM polish** pass to remove `unsafe` and improve idiomaticity
4. `cargo check` result is reported

---

## LLM configuration

Set one of the following in your environment or a `.env` file in the repo root:

```bash
# GitHub Copilot (default; token auto-detected from VS Code credential store)
LLM_PROVIDER=copilot
LLM_API_KEY=<your-github-oauth-token>

# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Any OpenAI-compatible endpoint (e.g. Ollama)
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1/chat/completions
LLM_API_KEY=<optional>
LLM_MODEL=llama3
```

The `.env` file is listed in `.gitignore` — **never commit API keys**.

---

## Output layout

```
output/
├── fortran/
│   └── blas/          # Downloaded BLAS .f source files
├── rust/
│   └── blas_converted/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── dgemm.rs
│           └── ...
└── reports/
    └── 20240101_120000_report.md
```

---

## Development

```bash
# Install in editable mode
pip install -e .

# Run the non-interactive smoke test
python convert.py --non-interactive

# Run the test suite (once tests are added)
# pytest
```

---

## Roadmap

- [x] Milestone 1 — Automate `dgemm` conversion (all three strategies)
- [ ] Milestone 2 — Extend to 10 BLAS functions
- [ ] Milestone 3 — Full BLAS library

---

## License

[MIT](LICENSE)
