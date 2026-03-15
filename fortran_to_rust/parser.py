"""Parse Fortran source files: extract subroutines/functions and their metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FortranRoutine:
    """Represents a single Fortran subroutine or function."""

    name: str
    kind: str  # 'subroutine' or 'function'
    source: str  # raw Fortran source text
    source_file: Optional[Path] = None
    args: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    line_count: int = 0
    chunks: List[str] = field(default_factory=list)  # ≤200-line pieces
    return_type: Optional[str] = None  # e.g. 'DOUBLE PRECISION' for typed functions

    def __post_init__(self) -> None:
        self.line_count = len(self.source.splitlines())
        if not self.chunks:
            self.chunks = _split_into_chunks(self.source)


def _split_into_chunks(source: str, max_lines: int = 200) -> List[str]:
    """Split source into chunks of at most max_lines lines."""
    lines = source.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return [source]
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunks.append("".join(lines[i : i + max_lines]))
    return chunks


# ---------------------------------------------------------------------------
# Fortran-77 / fixed-form regex helpers
# ---------------------------------------------------------------------------
_BLANK_OR_COMMENT = re.compile(r"^\s*$|^[Cc*!]")
_CONTINUATION = re.compile(r"^     [^ ]")  # col 6 non-blank = continuation

# Match SUBROUTINE or FUNCTION start — handles optional return-type prefix such as
# "DOUBLE PRECISION FUNCTION DDOT(...)" or "INTEGER FUNCTION IDAMAX(...)".
_ROUTINE_START = re.compile(
    r"^\s{0,6}"
    r"(?:(?P<rtype>"
    r"DOUBLE\s+PRECISION"
    r"|REAL(?:\s*\*\s*\d+)?"
    r"|INTEGER(?:\s*\*\s*\d+)?"
    r"|LOGICAL"
    r"|CHARACTER(?:\s*\*\s*\d+)?"
    r"|COMPLEX(?:\s*\*\s*\d+)?"
    r")\s+)?"
    r"(?:RECURSIVE\s+)?"
    r"(?P<kind>SUBROUTINE|FUNCTION)\s+(?P<name>\w+)\s*\(",
    re.IGNORECASE,
)
# Also match functions without parentheses (rare)
_ROUTINE_START_NOPAREN = re.compile(
    r"^\s{0,6}(?:RECURSIVE\s+)?(?P<kind>SUBROUTINE|FUNCTION)\s+(?P<name>\w+)\s*$",
    re.IGNORECASE,
)
_ROUTINE_END = re.compile(r"^\s{0,6}END\s*$", re.IGNORECASE)
_END_ROUTINE = re.compile(
    r"^\s{0,6}END\s+(?:SUBROUTINE|FUNCTION)\b", re.IGNORECASE
)
_CALL_STMT = re.compile(r"^\s+CALL\s+(\w+)\s*\(", re.IGNORECASE)
_FUNC_CALL = re.compile(r"\b([A-Z]\w+)\s*\(", re.IGNORECASE)
_ARG_LIST = re.compile(r"\(([^)]*)\)")


def _extract_args(header_line: str) -> List[str]:
    """Extract argument names from a SUBROUTINE/FUNCTION header line."""
    m = _ARG_LIST.search(header_line)
    if not m:
        return []
    return [a.strip() for a in m.group(1).split(",") if a.strip()]


def _extract_calls(source: str) -> List[str]:
    """Extract names of called subroutines/functions from source."""
    calls: set = set()
    for line in source.splitlines():
        # Explicit CALL statement
        m = _CALL_STMT.match(line)
        if m:
            calls.add(m.group(1).upper())
        # Function calls (heuristic: any word followed by '(')
        for m2 in _FUNC_CALL.finditer(line):
            name = m2.group(1).upper()
            # Filter out Fortran keywords / intrinsics
            if name not in _FORTRAN_KEYWORDS:
                calls.add(name)
    return sorted(calls)


_FORTRAN_KEYWORDS = {
    "IF", "DO", "WHILE", "THEN", "ELSE", "END", "RETURN", "WRITE", "READ",
    "FORMAT", "PRINT", "OPEN", "CLOSE", "MAX", "MIN", "ABS", "SQRT", "INT",
    "REAL", "DBLE", "ICHAR", "CHAR", "CMPLX", "CONJG", "AIMAG", "LOG",
    "EXP", "SIN", "COS", "TAN", "ATAN", "ATAN2", "SIGN", "MOD", "DIM",
    "PARAMETER", "IMPLICIT", "NONE", "INTEGER", "DOUBLE", "PRECISION",
    "LOGICAL", "CHARACTER", "EXTERNAL", "INTRINSIC", "DATA", "COMMON",
    "EQUIVALENCE", "SAVE", "ALLOCATE", "DEALLOCATE", "NULLIFY",
    "SUBROUTINE", "FUNCTION", "PROGRAM", "MODULE", "CONTAINS", "USE",
    "CALL", "GOTO", "CONTINUE", "STOP",
}


def parse_file(path: Path) -> List[FortranRoutine]:
    """Parse a Fortran source file and return a list of routines."""
    source = path.read_text(errors="replace")
    return parse_source(source, source_file=path)


def parse_source(
    source: str, source_file: Optional[Path] = None
) -> List[FortranRoutine]:
    """Parse Fortran source text and return a list of routines."""
    lines = source.splitlines(keepends=True)
    routines: List[FortranRoutine] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = _ROUTINE_START.match(line) or _ROUTINE_START_NOPAREN.match(line)
        if m:
            kind = m.group("kind").lower()
            name = m.group("name").upper()
            # Capture optional return type (e.g. "DOUBLE PRECISION" in
            # "DOUBLE PRECISION FUNCTION DDOT(...)").
            rtype_raw = m.group("rtype") if "rtype" in m.groupdict() else None
            return_type = " ".join(rtype_raw.split()).upper() if rtype_raw else None
            start = i
            args = _extract_args(line)
            # Collect until END (or END SUBROUTINE / END FUNCTION)
            i += 1
            while i < n:
                l = lines[i]
                if _END_ROUTINE.match(l) or _ROUTINE_END.match(l):
                    i += 1
                    break
                i += 1
            body = "".join(lines[start:i])
            calls = _extract_calls(body)
            routine = FortranRoutine(
                name=name,
                kind=kind,
                source=body,
                source_file=source_file,
                args=args,
                calls=calls,
                return_type=return_type,
            )
            routines.append(routine)
        else:
            i += 1

    # If nothing found, treat whole file as a single unnamed routine
    if not routines and source.strip():
        # Try to infer name from filename
        fname = source_file.stem.upper() if source_file else "UNKNOWN"
        routines.append(
            FortranRoutine(
                name=fname,
                kind="subroutine",
                source=source,
                source_file=source_file,
                calls=_extract_calls(source),
            )
        )

    return routines


def strip_comments(source: str) -> str:
    """Remove Fortran comment lines and trailing inline comments."""
    result = []
    for line in source.splitlines(keepends=True):
        # Full-line comments (cols 1, C, *, !)
        if re.match(r"^[Cc*!]", line):
            continue
        # Blank lines
        if line.strip() == "":
            continue
        # Inline ! comments (outside strings – simple heuristic)
        if "!" in line:
            # Don't strip if it's in a string literal (simple heuristic)
            idx = line.index("!")
            before = line[:idx]
            if before.count("'") % 2 == 0 and before.count('"') % 2 == 0:
                line = before.rstrip() + "\n"
        result.append(line)
    return "".join(result)


def get_implicit_types(source: str) -> Dict[str, str]:
    """Parse IMPLICIT statements and return letter→type mapping."""
    types: Dict[str, str] = {
        # Default Fortran-77 implicit typing: I-N are INTEGER, rest REAL
        chr(c): "REAL" for c in range(ord("A"), ord("Z") + 1)
    }
    for c in range(ord("I"), ord("N") + 1):
        types[chr(c)] = "INTEGER"

    implicit_re = re.compile(
        r"IMPLICIT\s+(\w+(?:\s+PRECISION)?)\s*\(([^)]+)\)", re.IGNORECASE
    )
    none_re = re.compile(r"IMPLICIT\s+NONE", re.IGNORECASE)

    for line in source.splitlines():
        if none_re.match(line.strip()):
            return {}
        m = implicit_re.search(line)
        if m:
            dtype = m.group(1).upper().replace(" ", "_")
            spec = m.group(2)
            for part in spec.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-")
                    for c in range(ord(a.strip().upper()), ord(b.strip().upper()) + 1):
                        types[chr(c)] = dtype
                else:
                    types[part.strip().upper()] = dtype
    return types
