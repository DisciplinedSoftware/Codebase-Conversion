"""Build a call graph from a list of parsed Fortran routines."""

from __future__ import annotations

from typing import Dict, List, Set

from fortran_to_rust.parser import FortranRoutine


def build_call_graph(routines: List[FortranRoutine]) -> Dict[str, List[str]]:
    """Return a mapping of routine_name -> [called_routine_names].

    Only edges to routines that exist in the *routines* list are kept.
    """
    known: Set[str] = {r.name.upper() for r in routines}
    graph: Dict[str, List[str]] = {}
    for routine in routines:
        edges = [c for c in routine.calls if c in known]
        graph[routine.name.upper()] = edges
    return graph


def topological_order(graph: Dict[str, List[str]]) -> List[str]:
    """Return routine names in topological order (leaves first).

    Routines that are called by others come before their callers so that
    helper functions are converted before the functions that use them.
    Uses Kahn's algorithm; cycles are broken arbitrarily.
    """
    from collections import deque

    in_degree: Dict[str, int] = {n: 0 for n in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[node] += 1

    queue: deque = deque(n for n, d in in_degree.items() if d == 0)
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for other, deps in graph.items():
            if node in deps:
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)

    # Append any remaining nodes (cycle members)
    remaining = [n for n in graph if n not in order]
    order.extend(remaining)
    return order


def render_graph(graph: Dict[str, List[str]]) -> str:
    """Return a simple text representation of the call graph."""
    lines = []
    for node in sorted(graph):
        deps = graph[node]
        if deps:
            lines.append(f"  {node} -> {', '.join(sorted(deps))}")
        else:
            lines.append(f"  {node} (leaf)")
    return "\n".join(lines)
