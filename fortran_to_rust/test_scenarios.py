"""Named test scenarios for exhaustive accuracy testing.

A :class:`TestScenario` parameterises one accuracy-test run.  Unlike the
plain sequential-seed tests in :func:`run_accuracy_check`, scenarios have
human-readable names and can override specific dimension values, scalar
floating-point arguments, and character (``CHARACTER*1``) arguments so that
boundary conditions and special cases are exercised systematically.

Both the Fortran reference and the converted Rust binary run **every** scenario
in the same order with the same numerical inputs, so any divergence in output
is immediately visible.

Scenario design principles
--------------------------
* **Random seeds** (``seed=0..4``): sanity-check normal operation across five
  independent input sets.
* **Dimension edge cases** (``N=0``, ``M=0``, ``K=0``): BLAS defines these as
  valid no-ops.  Both Fortran (via the standard quick-return) and Rust must
  agree on producing no output modifications.
* **Special scalar values** (``alpha=0``, ``beta=0``, ``beta=1``): these
  trigger distinct code paths in almost every BLAS routine.
* **Transpose variants**: ensure ``TRANSA='T'``/``TRANSB='T'`` paths are
  exercised for Level-3 routines.
* **Size variants**: small (1×1), medium (default ≈4×4), large (8×8 / 32
  elements) to surface numerical instability at scale.

Fortran error-handling note
----------------------------
Scenarios that pass *invalid* arguments (e.g. ``INCX=0``) deliberately
trigger the BLAS ``XERBLA`` / ``INFO`` convention.  The test harness compiles
those drivers with a *silent* XERBLA stub so the process does not abort, then
verifies that both Fortran and Rust leave the output arrays unchanged (the
correct behaviour after an invalid-argument early-return).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestScenario:
    """Parameters for one accuracy-test run.

    Attributes
    ----------
    name:
        Short identifier used in test output and file names.
    description:
        Human-readable explanation of what this scenario tests.
    seed:
        Seed for the random-input generator.  Changing the seed exercises
        different numerical values while keeping all other parameters fixed.
    dim_overrides:
        Override specific INTEGER scalar arguments (``N``, ``M``, ``K``,
        ``LDA``, ``LDB``, ``LDC``, ``INCX``, ``INCY``, …).  These replace
        the values computed by ``_assign_dims``.
    float_overrides:
        Override specific ``DOUBLE PRECISION`` scalar arguments by name
        (e.g. ``{"ALPHA": 0.0}``).  Applied *after* dataset generation so
        the override is guaranteed regardless of what the RNG produces.
    char_overrides:
        Override ``CHARACTER*1`` arguments (e.g. ``{"TRANSA": "T"}``).
        Applied to both the Fortran driver source and the Rust example source.
    expects_noop:
        When ``True`` the scenario is expected to produce no output (e.g.
        ``N=0`` quick-return or an invalid-argument early-return).  The test
        verifies that **both** Fortran and Rust agree: either both produce an
        empty output or both produce the same (unchanged) values.
    invalid_input:
        When ``True`` the inputs deliberately violate a BLAS precondition.
        The test harness compiles Fortran with a silent ``XERBLA`` stub and
        verifies that output arrays are unchanged in both implementations.
    """

    # Prevent pytest from treating this dataclass as a test class.
    __test__ = False

    name: str
    description: str
    seed: int = 0
    dim_overrides: Dict[str, int] = field(default_factory=dict)
    float_overrides: Dict[str, float] = field(default_factory=dict)
    char_overrides: Dict[str, str] = field(default_factory=dict)
    expects_noop: bool = False
    invalid_input: bool = False


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------

#: Five random-seed scenarios shared by every function class.
_RANDOM_SEEDS: List[TestScenario] = [
    TestScenario(
        f"random_seed_{i}",
        f"Random inputs, seed {i} — general numerical correctness",
        seed=i,
    )
    for i in range(5)
]


# ---------------------------------------------------------------------------
# BLAS Level 1  (vector-vector: DAXPY, DDOT, DSCAL, DCOPY, DSWAP, DASUM,
#                DNRM2, DROT, IDAMAX, …)
# ---------------------------------------------------------------------------

#: Exhaustive scenario list for BLAS Level-1 routines.
BLAS_LEVEL1_SCENARIOS: List[TestScenario] = [
    *_RANDOM_SEEDS,

    # --- Dimension edge cases ---
    TestScenario(
        "n_zero",
        "n=0 is a defined BLAS no-op; output must be unchanged",
        dim_overrides={"N": 0},
        expects_noop=True,
    ),
    TestScenario(
        "n_one",
        "n=1 single-element vectors",
        dim_overrides={"N": 1},
    ),
    TestScenario(
        "n_large",
        "n=32 longer vectors stress non-trivial loop body",
        dim_overrides={"N": 32},
    ),

    # --- Special scalar values ---
    TestScenario(
        "alpha_zero",
        "alpha=0 → DAXPY/DSCAL is a documented no-op for output",
        float_overrides={"DA": 0.0, "ALPHA": 0.0},
        expects_noop=True,
    ),
    TestScenario(
        "alpha_neg_one",
        "alpha=-1 → subtract DX from DY",
        float_overrides={"DA": -1.0, "ALPHA": -1.0},
    ),
    TestScenario(
        "alpha_large",
        "alpha=1000 → amplify numerical differences between implementations",
        float_overrides={"DA": 1000.0, "ALPHA": 1000.0},
    ),

    # --- Stride variants ---
    TestScenario(
        "stride_2",
        "incx=incy=2 reads/writes every other element",
        dim_overrides={"N": 4, "INCX": 2, "INCY": 2},
    ),
    TestScenario(
        "stride_3",
        "incx=incy=3 non-unit stride",
        dim_overrides={"N": 3, "INCX": 3, "INCY": 3},
    ),

    # --- Invalid-input scenarios (BLAS error-handling path) ---
    # Fortran: XERBLA is called, output unchanged.
    # Rust: must also return early without modifying output.
    TestScenario(
        "invalid_incx_zero",
        "INCX=0 is illegal; XERBLA must be called, output unchanged (Fortran INFO=4)",
        dim_overrides={"INCX": 0},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_incy_zero",
        "INCY=0 is illegal; XERBLA must be called, output unchanged (Fortran INFO=6)",
        dim_overrides={"INCY": 0},
        invalid_input=True,
        expects_noop=True,
    ),
]


# ---------------------------------------------------------------------------
# BLAS Level 2  (matrix-vector: DGEMV, DGER, DTRSV, DSYMV, …)
# ---------------------------------------------------------------------------

#: Exhaustive scenario list for BLAS Level-2 routines.
BLAS_LEVEL2_SCENARIOS: List[TestScenario] = [
    *_RANDOM_SEEDS,

    # --- Dimension edge cases ---
    TestScenario(
        "m_zero",
        "m=0 quick return; output unchanged",
        dim_overrides={"M": 0},
        expects_noop=True,
    ),
    TestScenario(
        "n_zero",
        "n=0 quick return; output unchanged",
        dim_overrides={"N": 0},
        expects_noop=True,
    ),
    TestScenario(
        "square_1x1",
        "1×1 matrix — minimal non-trivial case",
        dim_overrides={"M": 1, "N": 1, "LDA": 1},
    ),
    TestScenario(
        "square_8x8",
        "8×8 matrix — larger workload",
        dim_overrides={"M": 8, "N": 8, "LDA": 8},
    ),

    # --- Special scalar values ---
    TestScenario(
        "alpha_zero",
        "alpha=0 → no contribution from A*x",
        float_overrides={"ALPHA": 0.0},
    ),
    TestScenario(
        "beta_zero",
        "beta=0 → output vector zeroed before accumulation",
        float_overrides={"BETA": 0.0},
    ),
    TestScenario(
        "unit_beta",
        "beta=1 → pure accumulation (C += alpha*A*x)",
        float_overrides={"BETA": 1.0},
    ),
    TestScenario(
        "alpha_zero_beta_zero",
        "both zero → output must be zeroed",
        float_overrides={"ALPHA": 0.0, "BETA": 0.0},
    ),

    # --- Transpose variants ---
    TestScenario(
        "trans_n",
        "TRANS='N' (no transpose) — explicit check",
        char_overrides={"TRANS": "N"},
    ),
    TestScenario(
        "trans_t",
        "TRANS='T' (transpose)",
        char_overrides={"TRANS": "T"},
    ),

    # --- Invalid-input scenarios ---
    TestScenario(
        "invalid_m_neg",
        "M<0 is illegal; XERBLA must be called (Fortran INFO=1)",
        dim_overrides={"M": -1},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_n_neg",
        "N<0 is illegal; XERBLA must be called (Fortran INFO=2)",
        dim_overrides={"N": -1},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_lda_small",
        "LDA<max(1,M) is illegal; XERBLA must be called",
        dim_overrides={"LDA": 0},
        invalid_input=True,
        expects_noop=True,
    ),
]


# ---------------------------------------------------------------------------
# BLAS Level 3  (matrix-matrix: DGEMM, DSYMM, DTRSM, DSYRK, …)
# ---------------------------------------------------------------------------

#: Exhaustive scenario list for BLAS Level-3 routines.
BLAS_LEVEL3_SCENARIOS: List[TestScenario] = [
    *_RANDOM_SEEDS,

    # --- Dimension edge cases ---
    TestScenario(
        "m_zero",
        "m=0 quick return",
        dim_overrides={"M": 0},
        expects_noop=True,
    ),
    TestScenario(
        "n_zero",
        "n=0 quick return",
        dim_overrides={"N": 0},
        expects_noop=True,
    ),
    TestScenario(
        "k_zero",
        "k=0 → C := beta*C only (no A*B contribution)",
        dim_overrides={"K": 0},
    ),
    TestScenario(
        "square_1x1",
        "1×1×1 scalar multiply",
        dim_overrides={"M": 1, "N": 1, "K": 1, "LDA": 1, "LDB": 1, "LDC": 1},
    ),
    TestScenario(
        "square_8x8",
        "8×8×8 non-trivial matrix multiply",
        dim_overrides={"M": 8, "N": 8, "K": 8, "LDA": 8, "LDB": 8, "LDC": 8},
    ),
    TestScenario(
        "rectangular_4x6_k2",
        "Rectangular: C(4×6) = A(4×2) * B(2×6)",
        dim_overrides={"M": 4, "N": 6, "K": 2, "LDA": 4, "LDB": 2, "LDC": 4},
    ),

    # --- Special scalar values ---
    TestScenario(
        "alpha_zero",
        "alpha=0 → C := beta*C (A*B contribution suppressed)",
        float_overrides={"ALPHA": 0.0},
    ),
    TestScenario(
        "beta_zero",
        "beta=0 → C := alpha*A*B (no accumulation into prior C)",
        float_overrides={"BETA": 0.0},
    ),
    TestScenario(
        "unit_beta",
        "beta=1 → C += alpha*A*B",
        float_overrides={"BETA": 1.0},
    ),
    TestScenario(
        "alpha_zero_beta_zero",
        "both zero → C must be zeroed",
        float_overrides={"ALPHA": 0.0, "BETA": 0.0},
    ),
    TestScenario(
        "alpha_zero_unit_beta",
        "alpha=0, beta=1 → no-op (C is unchanged): verified quick-return",
        float_overrides={"ALPHA": 0.0, "BETA": 1.0},
        expects_noop=True,
    ),
    TestScenario(
        "alpha_neg_one",
        "alpha=-1 → subtract A*B from beta*C",
        float_overrides={"ALPHA": -1.0},
    ),

    # --- Transpose variants (square so LDA/LDB/LDC valid in all orientations) ---
    TestScenario(
        "trans_a",
        "op(A)=A^T, square 4×4×4",
        dim_overrides={"M": 4, "N": 4, "K": 4, "LDA": 4, "LDB": 4, "LDC": 4},
        char_overrides={"TRANSA": "T"},
    ),
    TestScenario(
        "trans_b",
        "op(B)=B^T, square 4×4×4",
        dim_overrides={"M": 4, "N": 4, "K": 4, "LDA": 4, "LDB": 4, "LDC": 4},
        char_overrides={"TRANSB": "T"},
    ),
    TestScenario(
        "trans_both",
        "op(A)=A^T, op(B)=B^T, square 4×4×4",
        dim_overrides={"M": 4, "N": 4, "K": 4, "LDA": 4, "LDB": 4, "LDC": 4},
        char_overrides={"TRANSA": "T", "TRANSB": "T"},
    ),

    # --- Invalid-input scenarios ---
    TestScenario(
        "invalid_transa",
        "TRANSA not in {N,T,C}: XERBLA INFO=1 (Fortran convention)",
        char_overrides={"TRANSA": "X"},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_m_neg",
        "M<0 illegal: XERBLA INFO=3",
        dim_overrides={"M": -1},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_n_neg",
        "N<0 illegal: XERBLA INFO=4",
        dim_overrides={"N": -1},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_k_neg",
        "K<0 illegal: XERBLA INFO=5",
        dim_overrides={"K": -1},
        invalid_input=True,
        expects_noop=True,
    ),
    TestScenario(
        "invalid_lda_small",
        "LDA too small: XERBLA INFO=8",
        dim_overrides={"LDA": 0},
        invalid_input=True,
        expects_noop=True,
    ),
]


# ---------------------------------------------------------------------------
# Function-name dispatch
# ---------------------------------------------------------------------------

# BLAS Level-1 function stems (lowercase, no leading 'd'/'s'/'c'/'z').
_LEVEL1_STEMS = {
    "daxpy", "saxpy", "zaxpy", "caxpy",
    "ddot",  "sdot",  "dsdot", "sdsdot",
    "dnrm2", "snrm2", "dznrm2", "scnrm2",
    "dasum", "sasum", "dzasum", "scasum",
    "dscal", "sscal", "zdscal", "zscal",
    "dcopy", "scopy", "zcopy", "ccopy",
    "dswap", "sswap", "zswap", "cswap",
    "drot",  "srot",  "zdrot", "csrot",
    "drotg", "srotg", "drotm", "srotm",
    "drotmg", "srotmg",
    "idamax", "isamax", "izamax", "icamax",
}

# BLAS Level-2 function stems.
_LEVEL2_STEMS = {
    "dgemv", "sgemv", "zgemv", "cgemv",
    "dgbmv", "sgbmv", "zgbmv", "cgbmv",
    "dsymv", "ssymv",
    "dsbmv", "ssbmv",
    "dspmv", "sspmv",
    "dtrmv", "strmv", "ztrmv", "ctrmv",
    "dtbmv", "stbmv", "ztbmv", "ctbmv",
    "dtpmv", "stpmv", "ztpmv", "ctpmv",
    "dtrsv", "strsv", "ztrsv", "ctrsv",
    "dtbsv", "stbsv", "ztbsv", "ctbsv",
    "dtpsv", "stpsv", "ztpsv", "ctpsv",
    "dger",  "sger",  "zgeru", "zgjerc", "cgeru", "cgerc",
    "dsyr",  "ssyr",  "zher",  "cher",
    "dspr",  "sspr",  "zhpr",  "chpr",
    "dsyr2", "ssyr2", "zher2", "cher2",
    "dspr2", "sspr2", "zhpr2", "chpr2",
}


def get_scenarios_for_function(function_name: str) -> List[TestScenario]:
    """Return the appropriate scenario list for *function_name*.

    Selection is based on the BLAS level inferred from the function name.
    Falls back to :data:`BLAS_LEVEL3_SCENARIOS` (the most comprehensive set)
    for unknown names so that newly-converted functions receive exhaustive
    coverage by default.
    """
    fn = function_name.lower()
    if fn in _LEVEL1_STEMS:
        return BLAS_LEVEL1_SCENARIOS
    if fn in _LEVEL2_STEMS:
        return BLAS_LEVEL2_SCENARIOS
    return BLAS_LEVEL3_SCENARIOS
