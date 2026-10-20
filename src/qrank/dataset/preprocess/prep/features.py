from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median_low


DEFAULT_FUNCTIONS = (
    "cumsum",
    "mean",
    "rank",
    "drop_duplicates",
    "round",
    "ffill",
    "shift",
    "sum",
)
FEATURE_KINDS = (
    "present",
    "count_bin",
    "missing_from_consensus",
    "extra_vs_consensus",
    "absolute_count_distance",
    "exact_consensus_match",
)
SYNTAX_ERROR_FEATURE = "syntax_error"


@dataclass(frozen=True)
class ImportTarget:
    library: str
    qualified_name: str


def imported_names(tree: ast.AST) -> dict[str, ImportTarget]:
    names = {
        # PrepBench's execution harness commonly provides these aliases.
        "pd": ImportTarget("pandas", "pandas"),
        "pandas": ImportTarget("pandas", "pandas"),
        "np": ImportTarget("numpy", "numpy"),
        "numpy": ImportTarget("numpy", "numpy"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".")[0]
                names[local] = ImportTarget(item.name.split(".")[0], item.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            library = node.module.split(".")[0]
            for item in node.names:
                if item.name != "*":
                    names[item.asname or item.name] = ImportTarget(
                        library,
                        f"{node.module}.{item.name}",
                    )
    return names


class PandasCallCounter(ast.NodeVisitor):
    """Conservatively follow dataframe lineage and count pandas calls."""

    def __init__(self, tree: ast.Module, imports: Mapping[str, ImportTarget]) -> None:
        self.imports = imports
        self.variable_libraries: defaultdict[str, set[str]] = defaultdict(set)
        self.calls: Counter[str] = Counter()
        self._infer_variables(tree)

    def expression_libraries(self, node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            target = self.imports.get(node.id)
            return {target.library} if target else set(self.variable_libraries[node.id])
        if isinstance(node, ast.Attribute):
            return self.expression_libraries(node.value)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "tables":
                return {"pandas"}
            return self.expression_libraries(node.value)
        if isinstance(node, ast.Call):
            return self.expression_libraries(node.func)
        if isinstance(node, ast.IfExp):
            return self.expression_libraries(node.body) | self.expression_libraries(
                node.orelse
            )
        return set()

    def _infer_variables(self, tree: ast.AST) -> None:
        assignments: list[tuple[ast.AST, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignments.extend((target, node.value) for target in node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                assignments.append((node.target, node.value))
        for _ in range(len(assignments) + 1):
            changed = False
            for target, value in assignments:
                libraries = self.expression_libraries(value)
                for name in self._assigned_names(target):
                    before = len(self.variable_libraries[name])
                    self.variable_libraries[name].update(libraries)
                    changed |= len(self.variable_libraries[name]) != before
            if not changed:
                break

    @staticmethod
    def _assigned_names(target: ast.AST) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [
                name
                for item in target.elts
                for name in PandasCallCounter._assigned_names(item)
            ]
        return []

    def visit_Call(self, node: ast.Call) -> None:
        if self.expression_libraries(node.func) == {"pandas"}:
            if isinstance(node.func, ast.Attribute):
                self.calls[node.func.attr] += 1
            elif isinstance(node.func, ast.Name):
                target = self.imports.get(node.func.id)
                if target and target.library == "pandas":
                    self.calls[target.qualified_name.rsplit(".", 1)[-1]] += 1
        self.generic_visit(node)


def pandas_function_counts(code: str) -> Counter[str]:
    tree = ast.parse(code)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    scope = ast.Module(body=functions, type_ignores=[])
    analyzer = PandasCallCounter(scope, imports=imported_names(tree))
    analyzer.visit(scope)
    return analyzer.calls


def feature_names(functions: Sequence[str] = DEFAULT_FUNCTIONS) -> tuple[str, ...]:
    return (
        *(
            f"pandas_{function}_{kind}"
            for function in functions
            for kind in FEATURE_KINDS
        ),
        SYNTAX_ERROR_FEATURE,
    )


def count_bin(count: int, cap: int) -> int:
    return min(max(int(count), 0), cap)


def build_feature_matrix(
    codes: Sequence[str],
    valid_mask: Iterable[bool],
    functions: Sequence[str] = DEFAULT_FUNCTIONS,
    *,
    count_bin_cap: int = 3,
) -> list[list[float]]:
    """Build aligned candidate features without consulting correctness labels."""

    if count_bin_cap < 1:
        raise ValueError("count_bin_cap must be at least 1")
    if not functions or len(functions) != len(set(functions)):
        raise ValueError("functions must be a non-empty unique sequence")
    valid = tuple(bool(value) for value in valid_mask)
    if len(codes) != len(valid):
        raise ValueError("codes and valid_mask must have the same length")

    counts: list[Counter[str]] = []
    syntax_errors: list[bool] = []
    for code in codes:
        try:
            counts.append(pandas_function_counts(code))
            syntax_errors.append(False)
        except SyntaxError:
            counts.append(Counter())
            syntax_errors.append(True)

    consensus_indices = [index for index, is_valid in enumerate(valid) if is_valid]
    if not consensus_indices:
        consensus_indices = list(range(len(codes)))
    consensus = {
        function: int(
            median_low(
                sorted(
                    count_bin(counts[index][function], count_bin_cap)
                    for index in consensus_indices
                )
            )
        )
        for function in functions
    }

    matrix: list[list[float]] = []
    for candidate_counts, syntax_error in zip(counts, syntax_errors):
        row: list[float] = []
        for function in functions:
            raw_count = candidate_counts[function]
            binned_count = count_bin(raw_count, count_bin_cap)
            expected = consensus[function]
            row.extend(
                [
                    float(raw_count > 0),
                    float(binned_count),
                    float(expected > 0 and raw_count == 0),
                    float(expected == 0 and raw_count > 0),
                    float(abs(binned_count - expected)),
                    float(binned_count == expected),
                ]
            )
        row.append(float(syntax_error))
        matrix.append(row)
    return matrix


__all__ = [
    "DEFAULT_FUNCTIONS",
    "FEATURE_KINDS",
    "SYNTAX_ERROR_FEATURE",
    "build_feature_matrix",
    "feature_names",
    "pandas_function_counts",
]
