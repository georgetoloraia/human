from typing import Dict, List, Tuple
from manager.mutations import (
    PATTERNS,
    MutationPattern,
    normalize_try_blocks,
    try_stats,
)


def propose_mutations(
    src: str, pattern_scores: Dict[str, float], error_scores: Dict[str, float]
) -> List[Tuple[str, str]]:
    """
    Return list of (new_code, pattern_name), ordered by combined scores.
    """
    normalized_src = normalize_try_blocks(src)
    base_try_counts, _ = try_stats(normalized_src)

    def combined_score(p: MutationPattern):
        base = pattern_scores.get(p.name, 0.5)
        err = error_scores.get(p.name, 0.5) if error_scores else 0.5
        return 0.7 * base + 0.3 * err

    ordered = sorted(PATTERNS, key=combined_score, reverse=True)
    proposals: List[Tuple[str, str]] = []
    seen_codes = set()
    for pattern in ordered:
        raw_new_code = pattern.apply(normalized_src)
        if not raw_new_code:
            continue
        new_code = normalize_try_blocks(raw_new_code)
        if new_code in seen_codes:
            continue

        candidate_try_counts, _ = try_stats(new_code)
        penalty = 0.0
        for fname, count in candidate_try_counts.items():
            base_count = base_try_counts.get(fname, 0)
            delta = count - base_count
            if delta > 0:
                penalty += delta * 0.5

        score = combined_score(pattern) - penalty
        proposals.append((score, new_code, pattern.name))
        seen_codes.add(new_code)

    proposals.sort(reverse=True, key=lambda x: x[0])
    return [(code, name) for (_score, code, name) in proposals]
