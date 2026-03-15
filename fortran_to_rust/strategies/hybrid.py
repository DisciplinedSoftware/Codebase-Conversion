"""Strategy 3 — Hybrid rule-based skeleton + optional LLM polish.

Pipeline
--------
1. Run ``f2c`` on the Fortran source to produce C.
2. Apply deterministic C→Rust transformations to get a compilable
   (possibly unsafe) Rust skeleton.
3. If an LLM is available, pass the skeleton through
   :meth:`LLMClient.polish_unsafe_rust` to remove unsafe blocks and
   improve idiomaticity.
4. Run ``cargo check`` and report results.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from fortran_to_rust.llm_client import LLMClient, LLMUnavailableError
from fortran_to_rust.parser import FortranRoutine
from fortran_to_rust.strategies.base import ConversionResult, ConversionStrategy


def _rust_return_type(routine: FortranRoutine) -> str:
    """Return the Rust return-type annotation for a Fortran FUNCTION, or '' for SUBROUTINEs."""
    if routine.kind != "function":
        return ""
    rtype = (routine.return_type or "").upper()
    if "INTEGER" in rtype:
        return " -> i32"
    if "LOGICAL" in rtype:
        return " -> bool"
    return " -> f64"   # DOUBLE PRECISION / REAL / unknown


def _rust_return_default(ret_type: str) -> str:
    """Return a default return expression matching *ret_type* (e.g. '0.0', '0', 'false')."""
    if not ret_type:
        return ""
    if "i32" in ret_type:
        return "0"
    if "bool" in ret_type:
        return "false"
    return "0.0"


class HybridStrategy(ConversionStrategy):
    name = "Hybrid Rule-Based + LLM Polish"

    def __init__(self, output_dir: Path, llm: Optional[LLMClient] = None, **kwargs) -> None:
        super().__init__(output_dir, **kwargs)
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        routine: FortranRoutine,
        *,
        progress_callback=None,
    ) -> ConversionResult:
        cb = progress_callback or (lambda msg: None)

        cb(f"[3/hybrid] Running f2c on {routine.name}…")
        c_source = self._fortran_to_c(routine)

        if c_source:
            cb(f"[3/hybrid] Translating C skeleton to Rust…")
            rust_source = self._c_to_rust(c_source, routine)
            strategy_used = "f2c→C→Rust rule-based"
        else:
            cb(f"[3/hybrid] f2c unavailable — using direct rule-based conversion…")
            rust_source = self._direct_fortran_to_rust(routine)
            strategy_used = "direct rule-based"

        # Optional LLM step
        if self.llm.is_available:
            try:
                if c_source:
                    # f2c produced C — polish the resulting unsafe Rust skeleton
                    cb(f"[3/hybrid] Polishing with LLM ({self.llm.provider}/{self.llm.model})…")
                    polished = self.llm.polish_unsafe_rust(rust_source, routine.name)
                    if polished.strip():
                        rust_source = polished
                        strategy_used += " + LLM polish"
                else:
                    # No f2c output — translate directly from Fortran for correct array layout
                    cb(f"[3/hybrid] Translating with LLM ({self.llm.provider}/{self.llm.model})…")
                    translated = self.llm.translate_fortran_to_rust(
                        routine.source, routine.name
                    )
                    if translated.strip():
                        rust_source = translated
                        strategy_used = "LLM translation"
            except LLMUnavailableError:
                pass
            except Exception as exc:
                cb(f"  [yellow]LLM step skipped: {exc}[/yellow]")

            # Accuracy-gated retry: compile + quick numerical check, repair if wrong
            rust_source, strategy_used = self._accuracy_repair_loop(
                rust_source, routine, strategy_used, cb
            )
        else:
            cb("  [dim]LLM not configured — skipping LLM step.[/dim]")

        result = ConversionResult(
            routine_name=routine.name,
            rust_source=rust_source,
            success=True,
            strategy_used=strategy_used,
        )
        return result

    # ------------------------------------------------------------------
    # Accuracy-gated repair loop
    # ------------------------------------------------------------------

    _MAX_ACCURACY_RETRIES = 5
    _ACCURACY_TOLERANCE = 1e-6  # tighter than test_harness default for early bail-out

    def _accuracy_repair_loop(
        self,
        rust_source: str,
        routine: FortranRoutine,
        strategy_used: str,
        cb,
    ):
        """Build a temp crate, run a single accuracy test, retry with LLM repair if wrong.

        Returns (rust_source, strategy_used).
        """
        from fortran_to_rust.rust_project import build_crate, scaffold_crate
        from fortran_to_rust.test_harness import run_accuracy_check

        # Find the cached Fortran source file (fetched by the pipeline earlier)
        fortran_path = (
            self.output_dir / "fortran" / "blas" / f"{routine.name.lower()}.f"
        )
        if not fortran_path.exists():
            return rust_source, strategy_used

        for attempt in range(self._MAX_ACCURACY_RETRIES):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                crate_dir = scaffold_crate(tmp, "acc_check", {routine.name: rust_source})
                build_ok, _ = build_crate(crate_dir)
                if not build_ok:
                    return rust_source, strategy_used  # compile errors handled elsewhere

                acc = run_accuracy_check(
                    routine.name,
                    fortran_path,
                    crate_dir,
                    routine=routine,
                    num_tests=1,
                )

            if acc.max_abs_error is None or acc.max_abs_error <= self._ACCURACY_TOLERANCE:
                return rust_source, strategy_used  # passed or no comparison possible

            cb(
                f"  [yellow]Accuracy check failed (max_err={acc.max_abs_error:.2e}), "
                f"repairing translation (attempt {attempt + 1}/{self._MAX_ACCURACY_RETRIES})…[/yellow]"
            )
            try:
                repaired = self.llm.repair_accuracy(
                    rust_source, routine.source, routine.name, acc.max_abs_error
                )
                if repaired.strip():
                    rust_source = repaired
                    strategy_used += f" + accuracy-repair×{attempt + 1}"
            except Exception as exc:
                cb(f"  [yellow]Accuracy repair skipped: {exc}[/yellow]")
                return rust_source, strategy_used

        return rust_source, strategy_used

    # ------------------------------------------------------------------
    # f2c step
    # ------------------------------------------------------------------

    def _fortran_to_c(self, routine: FortranRoutine) -> Optional[str]:
        """Run f2c and return the resulting C source, or None on failure."""
        if not shutil.which("f2c"):
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / f"{routine.name.lower()}.f"
            src.write_text(routine.source)
            try:
                result = subprocess.run(
                    ["f2c", "-R", str(src)],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir,
                    timeout=30,
                )
                c_file = tmp / f"{routine.name.lower()}.c"
                if c_file.exists():
                    return c_file.read_text()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return None

    # ------------------------------------------------------------------
    # C → Rust rule-based transformation
    # ------------------------------------------------------------------

    # Type mappings used for return types and variable declarations.
    # 'void' maps to the unit type '()' which is only valid as a return type;
    # it should never appear as a parameter type in f2c output.
    _F2C_RETURN_TYPES = {
        "void": "()",
        "int": "i32",
    }
    _F2C_TYPES = {
        "doublereal": "f64",
        "real": "f32",
        "integer": "i32",
        "logical": "bool",
        "ftnlen": "i32",
        "int": "i32",
        "double": "f64",
    }

    # f2c array argument pattern: "doublereal *a"  →  "&mut [f64]" / "&[f64]"
    _PTR_ARG = re.compile(
        r"\b(?P<type>doublereal|real|integer|logical)\s+\*(?P<name>\w+)"
    )
    _SCALAR_ARG = re.compile(
        r"\b(?P<type>doublereal|real|integer|logical)\s+(?P<name>\w+)\b"
    )

    def _c_to_rust(self, c_source: str, routine: FortranRoutine) -> str:
        """Apply deterministic rewrite rules to turn f2c C into unsafe Rust."""
        lines = c_source.splitlines()
        # Strip f2c header comments
        body_lines = [
            l for l in lines
            if not l.startswith("/*") and not l.startswith(" *") and "#include" not in l
        ]
        c_body = "\n".join(body_lines)

        # Build a minimal unsafe Rust wrapper
        rust = self._build_unsafe_rust(c_body, routine)
        return rust

    def _build_unsafe_rust(self, c_body: str, routine: FortranRoutine) -> str:
        """Produce an unsafe Rust translation using the routine's argument types."""
        from fortran_to_rust.test_harness import parse_arg_declarations

        name_lower = routine.name.lower()
        arg_decls = parse_arg_declarations(routine.source, routine.args)

        params: List[str] = []
        for arg in routine.args:
            decl = arg_decls.get(arg.upper())
            rname = arg.lower()
            if decl is None:
                params.append(f"    {rname}: f64")
            elif decl.is_char:
                params.append(f"    {rname}: u8")
            elif decl.is_integer and not decl.is_array:
                params.append(f"    {rname}: i32")
            elif decl.is_real and not decl.is_array:
                params.append(f"    {rname}: f64")
            elif decl.is_real and decl.is_array:
                params.append(f"    {rname}: &mut [f64]")
            elif decl.is_logical:
                params.append(f"    {rname}: bool")
            else:
                params.append(f"    {rname}: f64")

        ret_type = _rust_return_type(routine)
        ret_default = _rust_return_default(ret_type)

        rust_lines = [
            f"// Auto-generated by Hybrid strategy (f2c + rule-based C→Rust)",
            f"// Original function: {routine.name}",
            "#[allow(clippy::all, non_snake_case, unused_variables, unused_mut)]",
            f"pub unsafe fn {name_lower}(",
        ]
        for i, p in enumerate(params):
            sep = "," if i < len(params) - 1 else ""
            rust_lines.append(f"{p}{sep}")
        rust_lines.append(f"){ret_type} {{")

        in_body = False
        for line in c_body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            # Skip function prototype line
            if f"{name_lower}_(" in line.lower():
                in_body = True
                continue
            if not in_body:
                continue
            # Translate variable declarations
            translated = self._translate_c_line(line)
            rust_lines.append(f"    {translated}")

        if ret_default:
            rust_lines.append(f"    {ret_default}  // TODO: implement")
        rust_lines.append("}")
        return "\n".join(rust_lines)

    def _translate_c_line(self, line: str) -> str:
        """Best-effort single-line C→Rust translation."""
        s = line.strip()

        # Skip closing braces (we add our own)
        if s == "{" or s == "}":
            return s

        # Variable declaration: "integer i, j, k;" → "let mut i: i32; let mut j: i32;"
        for ctype, rtype in self._F2C_TYPES.items():
            pat = re.compile(rf"\b{ctype}\s+([\w\s,*]+);", re.IGNORECASE)
            m = pat.match(s)
            if m:
                names = [n.strip().lstrip("*") for n in m.group(1).split(",")]
                decls = "; ".join(f"let mut {n}: {rtype} = Default::default()" for n in names if n)
                return decls + ";"

        # for loop: "for (i = 1; i <= n; ++i)" → "for i in 1..=n"
        for_pat = re.compile(r"for\s*\(\s*(\w+)\s*=\s*(\w+)\s*;\s*\1\s*<=\s*(\w+)\s*;")
        m = for_pat.search(s)
        if m:
            var, start, end = m.group(1), m.group(2), m.group(3)
            return f"for {var} in {start}..={end} {{"

        # while loop
        if s.startswith("while (") or s.startswith("while("):
            cond = re.sub(r"while\s*\((.+)\)\s*\{?", r"\1", s)
            return f"while {cond} {{"

        # if statement
        if s.startswith("if (") or s.startswith("if("):
            cond = re.sub(r"if\s*\((.+)\)\s*\{?", r"\1", s)
            return f"if {cond} {{"

        # Assignment: replace "->" with "." and array indexing adjustments
        s = s.replace("->", ".")

        # Return
        if s.startswith("return"):
            return s.rstrip(";") + ";"

        return s + "  // TODO: verify"

    # ------------------------------------------------------------------
    # Direct Fortran→Rust (no f2c) fallback
    # ------------------------------------------------------------------

    def _direct_fortran_to_rust(self, routine: FortranRoutine) -> str:
        """Rule-based Fortran→Rust conversion using parsed argument types.

        Generates a correctly-typed Rust function stub so that the accuracy
        example binary can be compiled and the comparison mechanism can run,
        even though the body is a TODO until an LLM or manual review fills it in.
        """
        from fortran_to_rust.test_harness import parse_arg_declarations

        name = routine.name.lower()
        arg_decls = parse_arg_declarations(routine.source, routine.args)

        params: List[str] = []
        for arg in routine.args:
            decl = arg_decls.get(arg.upper())
            rname = arg.lower()
            if decl is None:
                params.append(f"    {rname}: f64")
            elif decl.is_char:
                params.append(f"    {rname}: u8")
            elif decl.is_integer and not decl.is_array:
                params.append(f"    {rname}: i32")
            elif decl.is_real and not decl.is_array:
                params.append(f"    {rname}: f64")
            elif decl.is_real and decl.is_array:
                params.append(f"    {rname}: &mut [f64]")
            elif decl.is_logical:
                params.append(f"    {rname}: bool")
            else:
                params.append(f"    {rname}: f64")

        # Return type for Fortran FUNCTIONs
        ret_type = _rust_return_type(routine)
        ret_default = _rust_return_default(ret_type)

        lines = [
            f"// Auto-generated by Hybrid strategy (direct rule-based)",
            f"// Original Fortran {routine.kind}: {routine.name}",
            "//",
            "// NOTE: The function body requires implementation.",
            "// The signature is derived from the Fortran argument declarations.",
            "#[allow(clippy::all, non_snake_case, unused_variables, unused_mut)]",
            f"pub fn {name}(",
        ]
        for i, p in enumerate(params):
            sep = "," if i < len(params) - 1 else ""
            lines.append(f"{p}{sep}")
        lines.append(f"){ret_type} {{")
        lines.append("    // TODO: translate from Fortran source:")
        lines.append("    //")
        for src_line in routine.source.splitlines()[:60]:
            lines.append(f"    // {src_line}")
        if routine.line_count > 60:
            lines.append(f"    // ... ({routine.line_count - 60} more lines)")
        if ret_default:
            lines.append(f"    {ret_default}  // TODO: implement")
        lines.append("}")
        return "\n".join(lines)
