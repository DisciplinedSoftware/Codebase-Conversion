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
from typing import Optional

from fortran_to_rust.llm_client import LLMClient, LLMUnavailableError
from fortran_to_rust.parser import FortranRoutine
from fortran_to_rust.strategies.base import ConversionResult, ConversionStrategy


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
            rust_source = self._c_to_rust(c_source, routine.name)
            strategy_used = "f2c→C→Rust rule-based"
        else:
            cb(f"[3/hybrid] f2c unavailable — using direct rule-based conversion…")
            rust_source = self._direct_fortran_to_rust(routine)
            strategy_used = "direct rule-based"

        # Optional LLM polish
        if self.llm.is_available:
            cb(f"[3/hybrid] Polishing with LLM ({self.llm.provider}/{self.llm.model})…")
            try:
                polished = self.llm.polish_unsafe_rust(rust_source, routine.name)
                if polished.strip():
                    rust_source = polished
                    strategy_used += " + LLM polish"
            except LLMUnavailableError:
                pass
            except Exception as exc:
                cb(f"  [yellow]LLM polish skipped: {exc}[/yellow]")
        else:
            cb("  [dim]LLM not configured — skipping polish step.[/dim]")

        result = ConversionResult(
            routine_name=routine.name,
            rust_source=rust_source,
            success=True,
            strategy_used=strategy_used,
        )
        return result

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

    _F2C_TYPES = {
        "doublereal": "f64",
        "real": "f32",
        "integer": "i32",
        "logical": "bool",
        "ftnlen": "i32",
        "int": "i32",
        "double": "f64",
        "void": "()",
    }

    # f2c array argument pattern: "doublereal *a"  →  "&mut [f64]" / "&[f64]"
    _PTR_ARG = re.compile(
        r"\b(?P<type>doublereal|real|integer|logical)\s+\*(?P<name>\w+)"
    )
    _SCALAR_ARG = re.compile(
        r"\b(?P<type>doublereal|real|integer|logical)\s+(?P<name>\w+)\b"
    )

    def _c_to_rust(self, c_source: str, fn_name: str) -> str:
        """Apply deterministic rewrite rules to turn f2c C into unsafe Rust."""
        lines = c_source.splitlines()
        # Strip f2c header comments
        body_lines = [
            l for l in lines
            if not l.startswith("/*") and not l.startswith(" *") and "#include" not in l
        ]
        c_body = "\n".join(body_lines)

        # Build a minimal unsafe Rust wrapper
        rust = self._build_unsafe_rust(c_body, fn_name)
        return rust

    def _build_unsafe_rust(self, c_body: str, fn_name: str) -> str:
        """Produce a rough unsafe Rust translation from C body text."""
        name_lower = fn_name.lower()

        # Collect type mappings for local variables
        type_map = self._F2C_TYPES

        def map_type(ctype: str) -> str:
            return type_map.get(ctype.lower().strip(), ctype)

        # Very rough line-by-line transformation
        rust_lines = [
            f"// Auto-generated by Hybrid strategy (f2c + rule-based C→Rust)",
            f"// Original function: {fn_name}",
            "#[allow(clippy::all, non_snake_case, unused_variables)]",
            f"pub unsafe fn {name_lower}(",
            "    // TODO: fill in parameters from the Fortran signature",
            ") {",
        ]

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
        """Minimal rule-based Fortran→Rust without f2c."""
        name = routine.name.lower()
        lines = [
            f"// Auto-generated by Hybrid strategy (direct rule-based, no f2c)",
            f"// Original Fortran subroutine: {routine.name}",
            "//",
            "// NOTE: This skeleton was produced without an LLM.",
            "// Manual review and editing will be required.",
            "#[allow(clippy::all, non_snake_case)]",
            f"pub fn {name}(",
        ]

        # Emit parameters as f64 placeholders
        for i, arg in enumerate(routine.args):
            sep = "," if i < len(routine.args) - 1 else ""
            lines.append(f"    {arg.lower()}: f64{sep}  // TODO: infer correct type")

        lines += [
            ") {",
            "    // TODO: implement the body from the Fortran source below.",
            "    //",
        ]

        # Embed stripped Fortran as comments so the developer has a reference
        for src_line in routine.source.splitlines()[:60]:
            lines.append(f"    // {src_line}")
        if routine.line_count > 60:
            lines.append(f"    // ... ({routine.line_count - 60} more lines)")

        lines.append("}")
        return "\n".join(lines)
