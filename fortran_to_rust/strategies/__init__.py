"""Conversion strategies for the Fortran-to-Rust pipeline."""

from fortran_to_rust.strategies.base import ConversionStrategy, ConversionResult
from fortran_to_rust.strategies.llm_first import LLMFirstStrategy
from fortran_to_rust.strategies.agentic import AgenticStrategy
from fortran_to_rust.strategies.hybrid import HybridStrategy

__all__ = [
    "ConversionStrategy",
    "ConversionResult",
    "LLMFirstStrategy",
    "AgenticStrategy",
    "HybridStrategy",
]

STRATEGY_MAP = {
    "1": LLMFirstStrategy,
    "2": AgenticStrategy,
    "3": HybridStrategy,
}

STRATEGY_NAMES = {
    "1": "LLM-First with Rule Fallback",
    "2": "Agentic Multi-Turn Dialogue",
    "3": "Hybrid Rule-Based + LLM Polish",
}
