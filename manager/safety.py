import ast


def is_safe(code: str) -> bool:
    """
    Basic safety: ensure code parses and is not excessively large.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    # Heuristic size cap to avoid runaway growth.
    node_count = sum(1 for _ in ast.walk(tree))
    return node_count < 5000
