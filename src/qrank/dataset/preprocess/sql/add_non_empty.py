"""Add candidate execution non-emptiness to a saved SQL DatasetDict.

Run with ``python -m qrank.dataset.preprocess.sql.add_non_empty DATASET DATABASES``.
The enriched dataset is saved beside DATASET with a ``_non_empty`` suffix.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from .builder import resolve_db_path
from .execution import execute_sql


def add_non_empty(dataset_path: Path, database_root: Path, timeout: float) -> Path:
    output_path = dataset_path.with_name(dataset_path.name + "_non_empty")
    if output_path.exists():
        raise FileExistsError(output_path)
    stored = load_from_disk(str(dataset_path))
    if not isinstance(stored, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {dataset_path}")

    def enrich(row: dict) -> dict:
        db_path = resolve_db_path(str(database_root), row["metadata"]["db_id"])
        candidates = []
        for candidate in row["candidates"]:
            non_empty = False
            if not candidate["invalid_mask"]:
                result = execute_sql(db_path, candidate["text"], timeout=timeout)
                non_empty = result.valid and bool(result.rows)
            candidates.append({
                **candidate,
                "is_non_empty_exec": non_empty,
            })
        return {"candidates": candidates}

    enriched = DatasetDict({
        name: split.map(enrich, desc=f"Checking {name} candidate results")
        for name, split in stored.items()
    })
    enriched.save_to_disk(str(output_path))
    if (dataset_path / "manifest.json").is_file():
        shutil.copyfile(dataset_path / "manifest.json", output_path / "manifest.json")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("databases", type=Path)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    print(add_non_empty(args.dataset, args.databases, args.timeout))


if __name__ == "__main__":
    main()
