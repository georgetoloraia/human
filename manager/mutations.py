import ast
import copy
from typing import Optional


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


class TryWrapPattern(MutationPattern):
    def __init__(self):
        super().__init__("try_wrap")

    def apply(self, src: str) -> Optional[str]:
        tree = ast.parse(src)
        changed = False
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
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
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.args.args:
                arg_name = node.args.args[0].arg
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


PATTERNS = [
    TryWrapPattern(),
    NoneGuardPattern(),
]
