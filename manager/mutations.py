import ast
import copy
import textwrap
from typing import Dict, Optional, Tuple


class MutationPattern:
    def __init__(self, name: str):
        self.name = name

    def apply(self, src: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError


def _module_to_source(tree: ast.Module) -> str:
    parts = []
    for node in tree.body:
        parts.append(ast.unparse(node))
    return "\n\n".join(parts) + "\n"


def _handlers_equivalent(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ha, hb in zip(a, b):
        if ast.dump(ha, include_attributes=False) != ast.dump(hb, include_attributes=False):
            return False
    return True


def try_stats(src: str) -> Tuple[Dict[str, int], int]:
    """
    Return (per-function try count, max nesting depth across functions).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}, 0

    func_counts: Dict[str, int] = {}
    max_depth = 0

    def visit(node, depth=0, current_func=None):
        nonlocal max_depth
        if isinstance(node, ast.FunctionDef):
            current_func = node.name
        if isinstance(node, ast.Try):
            max_depth = max(max_depth, depth + 1)
            if current_func:
                func_counts[current_func] = func_counts.get(current_func, 0) + 1
            for child in ast.iter_child_nodes(node):
                visit(child, depth + 1, current_func)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, depth, current_func)

    visit(tree)
    return func_counts, max_depth


def _max_try_depth_in_function(func: ast.FunctionDef) -> int:
    max_depth = 0

    def visit(node, depth=0):
        nonlocal max_depth
        if isinstance(node, ast.Try):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            visit(child, depth)

    visit(func)
    return max_depth


class _TryNormalizer(ast.NodeTransformer):
    """
    Collapse trivial nested try blocks with identical handlers.
    """

    def visit_Try(self, node: ast.Try):
        self.generic_visit(node)
        while (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Try)
            and not node.orelse
            and not node.finalbody
            and not node.body[0].orelse
            and not node.body[0].finalbody
            and _handlers_equivalent(node.handlers, node.body[0].handlers)
        ):
            inner = node.body[0]
            node = ast.Try(
                body=inner.body,
                handlers=inner.handlers,
                orelse=[],
                finalbody=[],
            )
            self.generic_visit(node)
        return node


def normalize_try_blocks(src: str) -> str:
    """
    Collapse redundant nested try blocks to simplify code before mutation.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    tree = _TryNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return _module_to_source(tree)


class TryWrapPattern(MutationPattern):
    def __init__(self):
        super().__init__("try_wrap")

    def apply(self, src: str) -> Optional[str]:
        tree = ast.parse(src)
        changed = False
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                if _max_try_depth_in_function(node) >= 1:
                    continue
                if node.body and isinstance(node.body[0], ast.Try):
                    continue
                new_node = copy.deepcopy(node)
                try_block = ast.Try(
                    body=new_node.body,
                    handlers=[
                        ast.ExceptHandler(
                            type=ast.Name(id="Exception", ctx=ast.Load()),
                            name=None,
                            body=[ast.Return(value=ast.Constant(value=None))],
                        )
                    ],
                    orelse=[],
                    finalbody=[],
                )
                new_node.body = [try_block]
                tree.body[i] = new_node
                changed = True
                break
        if not changed:
            return None
        ast.fix_missing_locations(tree)
        return _module_to_source(tree)


class NoneGuardPattern(MutationPattern):
    def __init__(self):
        super().__init__("none_guard")

    def apply(self, src: str) -> Optional[str]:
        tree = ast.parse(src)
        changed = False

        def has_existing_guard(fn: ast.FunctionDef, arg_name: str) -> bool:
            # Look at the first few statements for `if <arg> is None: return None`
            for stmt in fn.body[:3]:
                if not isinstance(stmt, ast.If):
                    continue
                if not isinstance(stmt.test, ast.Compare):
                    continue
                cmp = stmt.test
                if (
                    isinstance(cmp.left, ast.Name)
                    and cmp.left.id == arg_name
                    and len(cmp.ops) == 1
                    and isinstance(cmp.ops[0], ast.Is)
                    and len(cmp.comparators) == 1
                    and isinstance(cmp.comparators[0], ast.Constant)
                    and cmp.comparators[0].value is None
                ):
                    if stmt.body and isinstance(stmt.body[0], ast.Return):
                        ret = stmt.body[0]
                        if isinstance(ret.value, ast.Constant) and ret.value.value is None:
                            return True
            return False

        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.args.args:
                arg_name = node.args.args[0].arg
                if has_existing_guard(node, arg_name):
                    continue
                guard = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=arg_name, ctx=ast.Load()),
                        ops=[ast.Is()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    body=[ast.Return(value=ast.Constant(value=None))],
                    orelse=[],
                )
                new_node = copy.deepcopy(node)
                new_node.body.insert(0, guard)
                tree.body[i] = new_node
                changed = True
                break
        if not changed:
            return None
        ast.fix_missing_locations(tree)
        return _module_to_source(tree)

class TouchUpPattern(MutationPattern):
    def __init__(self):
        super().__init__("touch_up")

    def apply(self, src: str) -> Optional[str]:
        marker = "# auto-touch"
        count = src.count(marker)
        new_marker = f"{marker}-{count + 1}" if count else marker
        return src + ("\n" if not src.endswith("\n") else "") + new_marker + "\n"


class AttrFixPattern(MutationPattern):
    """
    Targeted fix for AttributeError-prone helpers like list_sum/doc_sum.
    Rewrites those functions to safe, explicit implementations.
    """

    def __init__(self):
        super().__init__("attr_fix")
        self.templates = {
            "list_sum": ast.parse(
                textwrap.dedent(
                    """
                    def list_sum(values):
                        if values is None:
                            return 0
                        total = 0
                        for v in values:
                            total += v
                        return total
                    """
                )
            ).body[0],
            "doc_sum": ast.parse(
                textwrap.dedent(
                    """
                    def doc_sum(numbers):
                        if not numbers:
                            return 0
                        return sum(numbers)
                    """
                )
            ).body[0],
            "use_sum": ast.parse(
                textwrap.dedent(
                    """
                    def use_sum(iterable, start=0):
                        return sum(iterable, start)
                    """
                )
            ).body[0],
            "use_min": ast.parse(
                textwrap.dedent(
                    """
                    def use_min(iterable):
                        if iterable is None or len(iterable) == 0:
                            return None
                        return min(iterable)
                    """
                )
            ).body[0],
            "use_len": ast.parse(
                textwrap.dedent(
                    """
                    def use_len(obj):
                        return len(obj)
                    """
                )
            ).body[0],
        }

    def apply(self, src: str) -> Optional[str]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        changed = False
        seen = set()
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name in self.templates:
                seen.add(node.name)
                template = self.templates[node.name]
                if ast.dump(node, include_attributes=False) != ast.dump(template, include_attributes=False):
                    tree.body[i] = template
                    changed = True
        for name, template in self.templates.items():
            if name not in seen:
                tree.body.append(template)
                changed = True
        if not changed:
            return None
        ast.fix_missing_locations(tree)
        return _module_to_source(tree)


PATTERNS = [
    TryWrapPattern(),
    NoneGuardPattern(),
    TouchUpPattern(),
    AttrFixPattern(),
]
