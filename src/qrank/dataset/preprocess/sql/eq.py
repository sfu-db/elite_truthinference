from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution import ExecutionResult


def _normalize_value(value):
    if value is None:
        return "<NULL>"
    return repr(value)


def _normalize_rows(rows):
    """
    Normalize query result rows.
    Keeps duplicate rows.
    Sorts rows for order-insensitive comparison.
    """

    if isinstance(rows, str):
        return rows

    return tuple(
        sorted(
            tuple(_normalize_value(v) for v in row)
            for row in rows
        )
    )


def results_equal(
    left: ExecutionResult,
    right: ExecutionResult,
    *,
    set_semantic: bool = True,
) -> bool:
    if not left.valid or not right.valid:
        return False
    left_norm = _normalize_rows(left.rows)
    right_norm = _normalize_rows(right.rows)
    if set_semantic:
        return set(left_norm) == set(right_norm)
    return left_norm == right_norm


def results_equal_all(
    left: Sequence[ExecutionResult],
    right: Sequence[ExecutionResult],
    *,
    set_semantic: bool = True,
) -> bool:
    """Whether two queries agree on every database; one result per database, in the same order."""
    return all(
        results_equal(left_db, right_db, set_semantic=set_semantic)
        for left_db, right_db in zip(left, right, strict=True)
    )


def build_agreement(
    results: Sequence[Sequence[ExecutionResult]],
    *,
    set_semantic: bool = True,
) -> list[list[int]]:
    """Pairwise equivalence of candidates, given each candidate's results on every database."""
    agreement: list[list[int]] = []
    for left in results:
        row: list[int] = []
        for right in results:
            eq = results_equal_all(left, right, set_semantic=set_semantic)
            row.append(int(eq))
        agreement.append(row)
    return agreement
