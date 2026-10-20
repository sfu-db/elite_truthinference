from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from .artifacts import DEFAULT_SETTING, PrepBenchArtifacts
from .eq import build_agreement, load_execution_result
from .features import DEFAULT_FUNCTIONS, build_feature_matrix, feature_names


class PrepBenchDatasetBuilder:
    """Build canonical QRank rows from retained PrepBench executions."""

    def __init__(
        self,
        package_root: str | Path,
        prepbench_root: str | Path,
        *,
        models: Sequence[str] | None = None,
        functions: Sequence[str] = DEFAULT_FUNCTIONS,
        count_bin_cap: int = 3,
        ignore_row_order: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        setting: str = DEFAULT_SETTING,
    ) -> None:
        self.artifacts = PrepBenchArtifacts(
            package_root,
            prepbench_root,
            setting=setting,
        )
        self.models = self.artifacts.resolve_models(models)
        self.functions = tuple(functions)
        if not self.functions or len(self.functions) != len(set(self.functions)):
            raise ValueError("functions must be a non-empty unique sequence")
        if count_bin_cap < 1:
            raise ValueError("count_bin_cap must be at least 1")
        if rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be non-negative")
        self.count_bin_cap = count_bin_cap
        self.ignore_row_order = ignore_row_order
        self.rtol = rtol
        self.atol = atol

    def process_case(self, case_id: str) -> dict[str, Any]:
        case_id = str(case_id)
        if case_id not in self.artifacts.case_ids:
            raise ValueError(f"Unknown case_id: {case_id}")

        artifacts = [
            self.artifacts.artifact(case_id, model) for model in self.models
        ]
        codes = [
            artifact.code_path.read_text(encoding="utf-8").strip()
            for artifact in artifacts
        ]
        if any(not code for code in codes):
            empty_models = [
                model for model, code in zip(self.models, codes) if not code
            ]
            raise ValueError(
                f"{case_id}: empty generated program for {', '.join(empty_models)}"
            )

        execution_results = [
            load_execution_result(
                artifact.case_path,
                ignore_row_order=self.ignore_row_order,
            )
            for artifact in artifacts
        ]
        valid_mask = [result.valid for result in execution_results]
        features = build_feature_matrix(
            codes,
            valid_mask,
            self.functions,
            count_bin_cap=self.count_bin_cap,
        )
        labels = [artifact.label for artifact in artifacts]
        aligned_labels = labels if all(label is not None for label in labels) else None

        candidate_rows = [
            {
                "source": model,
                "text": code,
                "features": candidate_features,
                "invalid_mask": not result.valid,
            }
            for model, code, candidate_features, result in zip(
                self.models,
                codes,
                features,
                execution_results,
            )
        ]
        return {
            "question_id": case_id,
            "question": self.artifacts.question(case_id),
            "candidates": candidate_rows,
            "feature_names": list(feature_names(self.functions)),
            "agreement": build_agreement(
                execution_results,
                rtol=self.rtol,
                atol=self.atol,
            ),
            "context": "",
            "evidence": None,
            "group_id": "prepbench",
            "metadata": {
                "dataset_type": "data-preparation-code",
                "setting": self.artifacts.setting,
                "case_id": case_id,
                "source_paths": [
                    artifact.relative_code_path for artifact in artifacts
                ],
                "invalid_reasons": [
                    result.error or "" for result in execution_results
                ],
                "comparison": {
                    "row_order_sensitive": not self.ignore_row_order,
                    "column_order_sensitive": False,
                    "rtol": self.rtol,
                    "atol": self.atol,
                    "correctness_labels_used": False,
                },
            },
            "labels": aligned_labels,
        }

    def build(self, case_ids: Sequence[str] | None = None) -> Dataset:
        selected = self.artifacts.case_ids if case_ids is None else tuple(map(str, case_ids))
        if not selected:
            raise ValueError("case_ids must not be empty")
        if len(selected) != len(set(selected)):
            raise ValueError("case_ids must be unique")
        unknown = [case_id for case_id in selected if case_id not in self.artifacts.case_ids]
        if unknown:
            raise ValueError("Unknown case_id(s): " + ", ".join(unknown))
        return Dataset.from_list([self.process_case(case_id) for case_id in selected])

    def build_splits(
        self,
        splits: Mapping[str, Sequence[str]],
    ) -> DatasetDict:
        return DatasetDict(
            {name: self.build(case_ids) for name, case_ids in splits.items()}
        )


__all__ = ["PrepBenchDatasetBuilder"]
