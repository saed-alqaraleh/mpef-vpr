#!/usr/bin/env python3
"""Reject prohibited budget-branch identifiers from production code."""

from __future__ import annotations

from pathlib import Path


FORBIDDEN = (
    "EntropyBudget",
    "allocate_budget",
    "matching_value",
    "budget_fidelity",
    "LOEFO",
    "frozen_universal_fusion",
)


def main() -> int:
    roots = [Path("src"), Path("scripts"), Path("configs"), Path("freezes")]
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.name == Path(__file__).name
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            text = path.read_text(errors="ignore")
            for token in FORBIDDEN:
                if token in text:
                    violations.append(f"{path}: {token}")
    if violations:
        print("\n".join(violations))
        return 1
    print("independence firewall OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
