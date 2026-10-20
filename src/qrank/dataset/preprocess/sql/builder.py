from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from .eq import build_agreement, results_equal_all
from .execution import ExecutionResult, execute_sql
from .features import QueryFeatures


def resolve_db_path(database_root: str, db_id: str) -> Path:
    path = Path(database_root) / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    return path


def resolve_instance_paths(instance_root: str, db_id: str, question_id: str, layout: str = "question") -> list[Path]:
    """The generated test databases of a question: ``<instance_root>/<db_id>/<question_id>/*.sqlite`` (ParSEval,
    layout "question") or ``<instance_root>/<db_id>/*.sqlite``, shared by the questions of a database (the Spider
    test suite, layout "database")."""
    if layout == "question":
        paths = sorted((Path(instance_root) / db_id / question_id).glob("*.sqlite"))
    elif layout == "database":
        paths = sorted((Path(instance_root) / db_id).glob("*.sqlite"))
    else:
        raise ValueError(f"Unknown instance layout: {layout}")
    if not paths:
        raise FileNotFoundError(f"No generated databases for question {question_id} in {instance_root}/{db_id}")
    return paths


def load_preprocessed(instance_root: str) -> dict[tuple[str, str], str | None]:
    """The SQL each query was given to ParSEval as, from the manifest beside the
    instances (``<instance_root>.jsonl``), keyed by (question_id, sql)."""
    manifest = Path(f"{instance_root}.jsonl")
    return {
        (str(record["question_id"]), record["sql"]): record["preprocessed_sql"]
        for record in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())
    }


class SQLDatasetBuilder:
    _REQUIRED_COLUMNS = frozenset(
        {"question_id", "question", "db_id", "models", "candidates"}
    )

    def __init__(
        self,
        database_root: str,
        *,
        dialect: str = "sqlite",
        timeout: float = 30.0,
        set_semantic: bool = True,
        instance_root: str | None = None,
        instance_layout: str = "question",
    ) -> None:
        """Two queries are equivalent (agreement and labels) when they give
        identical results on the original database and, with ``instance_root``,
        on every generated database of their question."""
        self.database_root = database_root
        self.dialect = dialect
        self.timeout = timeout
        self.set_semantic = set_semantic
        self.instance_root = instance_root
        self.instance_layout = instance_layout
        # Only ParSEval's instances come with the SQL each query was given as.
        self.preprocessed = None if instance_root is None or instance_layout != "question" else load_preprocessed(instance_root)

    def process_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        question_id = str(row["question_id"])
        question = str(row["question"])
        db_id = str(row["db_id"])

        models = [str(model) for model in row["models"]]
        candidates = [str(candidate) for candidate in row["candidates"]]
        if len(models) != len(candidates):
            raise ValueError(
                f"{question_id}: models and candidates must have the same length"
            )
        if not candidates:
            raise ValueError(f"{question_id}: candidates must not be empty")

        db_path = resolve_db_path(self.database_root, db_id)
        databases = [db_path]
        if self.instance_root is not None:
            databases += resolve_instance_paths(self.instance_root, db_id, question_id, self.instance_layout)
        cache: dict[tuple[Path, str], ExecutionResult] = {}

        def execute(path: Path, sql: str) -> ExecutionResult:
            if (path, sql) not in cache:
                cache[path, sql] = execute_sql(
                    path,
                    sql,
                    timeout=self.timeout,
                    dialect=self.dialect,
                )
            return cache[path, sql]

        # Validity and emptiness come from the original database.
        execution_results = [execute(db_path, sql) for sql in candidates]
        pooled_results = [[execute(path, sql) for path in databases] for sql in candidates]

        gold = row.get("gold")
        labels = None
        if gold is not None:
            gold_results = [execute(path, gold) for path in databases]
            labels = [
                results_equal_all(
                    gold_results,
                    results,
                    set_semantic=self.set_semantic,
                )
                for results in pooled_results
            ]

        features = [
            QueryFeatures.from_sql(sql, dialect=self.dialect)
            for sql in candidates
        ]
        agreement = build_agreement(
            pooled_results,
            set_semantic=self.set_semantic,
        )

        candidate_rows = [
            {
                "source": str(model),
                "text": candidate,
                "features": list(feature.values()),
                "invalid_mask": not result.valid or not candidate.strip(),
                "is_non_empty_exec": result.valid and bool(result.rows),
                # Invalid predictions were not given to ParSEval, so they have none.
                "preprocessed_text": None if self.preprocessed is None else self.preprocessed.get((question_id, candidate)),
            }
            for model, candidate, result, feature in zip(
                models, candidates, execution_results, features
            )
        ]

        return {
            "question_id": question_id,
            "question": question,
            "gold": None if gold is None else str(gold),
            "candidates": candidate_rows,
            "feature_names": list(QueryFeatures.feature_names()),
            "agreement": agreement,
            "context": str(row.get("context") or ""),
            "evidence": row.get("evidence"),
            "group_id": db_id,
            "metadata": {
                "dataset_type": "text-to-sql",
                "db_id": db_id,
                "difficulty": str(row.get("difficulty", "unknown")),
            },
            "labels": labels,
        }

    def build(self, frame: pd.DataFrame) -> Dataset:
        """Build a dataset from all rows in a pandas DataFrame.

        Pandas owns the frame transformation. SQL execution and parsing remain
        scalar operations because each row can target a different database.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.index.name == "question_id":
            frame = frame.reset_index()

        missing = self._REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"frame is missing required columns: {missing_columns}")
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        dataset = dataset.map(
            self.process_row,
            remove_columns=dataset.column_names,
            desc="Executing SQL and building features",
        )
        return dataset

    def build_splits(
        self,
        splits: Mapping[str, pd.DataFrame],
    ) -> DatasetDict:
        return DatasetDict(
            {name: self.build(frame) for name, frame in splits.items()}
        )


__all__ = ["SQLDatasetBuilder", "load_preprocessed", "resolve_db_path", "resolve_instance_paths"]
