"""Base class and result type shared by all conversion strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from fortran_to_rust.parser import FortranRoutine


@dataclass
class ConversionResult:
    """Outcome of converting a single Fortran routine."""

    routine_name: str
    rust_source: Optional[str] = None       # generated Rust code
    success: bool = False
    strategy_used: str = ""                 # human label for the method used
    error: Optional[str] = None            # error message on failure
    repair_rounds: int = 0                  # number of LLM repair iterations
    compiler_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_path: Optional[Path] = None      # path where .rs file was written


class ConversionStrategy(ABC):
    """Abstract base class for a Fortran-to-Rust conversion strategy."""

    #: Human-readable name shown in the CLI
    name: str = "Unknown Strategy"

    def __init__(self, output_dir: Path, **kwargs) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def convert(
        self,
        routine: FortranRoutine,
        *,
        progress_callback=None,
    ) -> ConversionResult:
        """Convert *routine* and return a :class:`ConversionResult`.

        *progress_callback*, if provided, is a ``callable(message: str)``
        used to emit status messages to the CLI.
        """
