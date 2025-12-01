from typing import Dict, List, Tuple
from manager.mutations import PATTERNS, MutationPattern


def propose_mutations(src: str, pattern_scores: Dict[str, float]) -> List[Tuple[str, str]]:
    """
    Return list of (new_code, pattern_name), ordered by learned pattern scores.
    """
    ordered = sorted(
        PATTERNS,
        key=lambda p: pattern_scores.get(p.name, 0.5),
        reverse=True,
    )
    proposals: List[Tuple[str, str]] = []
    seen_codes = set()
    for pattern in ordered:
        new_code = pattern.apply(src)
        if new_code and new_code not in seen_codes:
            proposals.append((new_code, pattern.name))
            seen_codes.add(new_code)
    return proposals
