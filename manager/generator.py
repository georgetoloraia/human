from typing import Dict, List, Tuple
from mutations import PATTERNS, MutationPattern


def propose_mutations(
    src: str, pattern_scores: Dict[str, float], error_scores: Dict[str, float]
) -> List[Tuple[str, str]]:
    """
    Return list of (new_code, pattern_name), ordered by combined scores.
    """
    def combined_score(p: MutationPattern):
        base = pattern_scores.get(p.name, 0.5)
        err = error_scores.get(p.name, 0.5) if error_scores else 0.5
        return 0.7 * base + 0.3 * err

    ordered = sorted(PATTERNS, key=combined_score, reverse=True)
    proposals: List[Tuple[str, str]] = []
    seen_codes = set()
    for pattern in ordered:
        new_code = pattern.apply(src)
        if new_code and new_code not in seen_codes:
            proposals.append((new_code, pattern.name))
            seen_codes.add(new_code)
    return proposals
