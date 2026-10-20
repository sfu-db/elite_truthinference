from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
from pandas.testing import assert_frame_equal


@dataclass
class ExecutionResult:
    """Retained outputs from one previously executed PrepBench candidate."""

    valid: bool
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    error: str | None = None


def normalize_frame(
    frame: pd.DataFrame,
    *,
    ignore_row_order: bool = True,
) -> pd.DataFrame:
    """Return a deterministic frame while preserving values and duplicates."""

    normalized = frame.reindex(columns=sorted(frame.columns)).reset_index(drop=True)
    if ignore_row_order and len(normalized) > 1:
        sort_keys = normalized.astype("string").fillna("\0")
        order = sort_keys.sort_values(
            by=list(sort_keys.columns),
            kind="mergesort",
        ).index
        normalized = normalized.loc[order]
    return normalized.reset_index(drop=True)


def _read_output_csv(path: Path) -> pd.DataFrame:
    """Read CSVs without losing integer identifiers larger than int64."""

    frame = pd.read_csv(path, dtype=str)
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        try:
            numeric = pd.to_numeric(values, errors="raise")
        except (ValueError, TypeError, OverflowError):
            continue
        if not pd.api.types.is_numeric_dtype(numeric.dtype):
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_execution_result(
    case_path: str | Path,
    *,
    execution_file: str = "solution/execution.json",
    output_subdirectory: str = "solution/cand",
    ignore_row_order: bool = True,
) -> ExecutionResult:
    """Load outputs produced by PrepBench without executing candidate code."""

    case_path = Path(case_path)
    execution_path = case_path / execution_file
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ExecutionResult(
            valid=False,
            error=f"execution metadata: {type(exc).__name__}: {exc}",
        )
    if not execution.get("ok", False):
        return ExecutionResult(valid=False, error="execution failed")

    output_dir = case_path / output_subdirectory
    csv_paths = sorted(path for path in output_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        return ExecutionResult(valid=False, error=f"no CSV outputs in {output_dir}")

    frames: dict[str, pd.DataFrame] = {}
    try:
        for path in csv_paths:
            frames[path.name] = normalize_frame(
                _read_output_csv(path),
                ignore_row_order=ignore_row_order,
            )
    except (OSError, ValueError, OverflowError, pd.errors.ParserError) as exc:
        return ExecutionResult(
            valid=False,
            error=f"CSV read: {type(exc).__name__}: {exc}",
        )
    return ExecutionResult(valid=True, frames=frames)


def frames_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    try:
        assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            check_categorical=False,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError:
        return False
    return True


def results_equal(
    left: ExecutionResult,
    right: ExecutionResult,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    if not left.valid or not right.valid:
        return False
    if left.frames.keys() != right.frames.keys():
        return False
    return all(
        frames_equal(left.frames[name], right.frames[name], rtol=rtol, atol=atol)
        for name in left.frames
    )


def build_agreement(
    results: Sequence[ExecutionResult],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[list[int]]:
    agreement: list[list[int]] = []
    for left in results:
        agreement.append(
            [
                int(results_equal(left, right, rtol=rtol, atol=atol))
                for right in results
            ]
        )
    return agreement


__all__ = [
    "ExecutionResult",
    "build_agreement",
    "frames_equal",
    "load_execution_result",
    "normalize_frame",
    "results_equal",
]
