"""
Simple evaluator stub. Extend with domain-specific checks to decide if a mutation should survive.
"""


def evaluate(original: str, mutated: str) -> bool:
    # Accept mutations by default; customize to enforce quality.
    return True
