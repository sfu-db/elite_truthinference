from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SETTING = "Prep-Code Generation"


def _parse_optional_bool(value: str, *, context: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{context}: expected true or false, got {value!r}")


@dataclass(frozen=True)
class CandidateArtifact:
    case_id: str
    model: str
    case_path: Path
    code_path: Path
    relative_code_path: str
    label: bool | None


class PrepBenchArtifacts:
    """Index the retained PrepBench package and its benchmark questions."""

    def __init__(
        self,
        package_root: str | Path,
        prepbench_root: str | Path,
        *,
        setting: str = DEFAULT_SETTING,
    ) -> None:
        self.package_root = Path(package_root)
        self.prepbench_root = Path(prepbench_root)
        self.setting = setting
        self._artifacts, self._models, self._case_ids = self._read_manifest()

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self._case_ids

    def resolve_models(self, requested: Sequence[str] | None) -> tuple[str, ...]:
        if requested is None:
            return self.models
        selected = tuple(str(model) for model in requested)
        if not selected:
            raise ValueError("models must not be empty")
        if len(selected) != len(set(selected)):
            raise ValueError("models must be unique")
        unknown = [model for model in selected if model not in self.models]
        if unknown:
            raise ValueError(
                "Unknown model(s): "
                + ", ".join(unknown)
                + ". Available: "
                + ", ".join(self.models)
            )
        return selected

    def artifact(self, case_id: str, model: str) -> CandidateArtifact:
        key = (str(case_id), str(model))
        try:
            artifact = self._artifacts[key]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing manifest artifact for {case_id}/{model}") from exc
        if not artifact.code_path.is_file():
            raise FileNotFoundError(
                f"Missing generated program for {case_id}/{model}: {artifact.code_path}"
            )
        return artifact

    def question(self, case_id: str) -> str:
        path = self.prepbench_root / "data" / str(case_id) / "query.md"
        if not path.is_file():
            raise FileNotFoundError(f"Missing PrepBench question: {path}")
        question = path.read_text(encoding="utf-8").strip()
        if not question:
            raise ValueError(f"Empty PrepBench question: {path}")
        return question

    def _read_manifest(
        self,
    ) -> tuple[
        dict[tuple[str, str], CandidateArtifact],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        manifest_path = self.package_root / "manifest.csv"
        artifacts: dict[tuple[str, str], CandidateArtifact] = {}
        models: list[str] = []
        case_ids: list[str] = []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "paper_setting",
                "model_slug",
                "case_id",
                "relative_case_path",
                "generated_artifact_file",
                "paper_metric_passed",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{manifest_path}: missing columns {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, 2):
                if row["paper_setting"] != self.setting:
                    continue
                model = row["model_slug"].strip()
                case_id = row["case_id"].strip()
                relative_case_path = row["relative_case_path"].strip()
                code_file = row["generated_artifact_file"].strip()
                context = f"{manifest_path}:{line_number}"
                if not all((model, case_id, relative_case_path, code_file)):
                    raise ValueError(f"{context}: incomplete PrepBench artifact row")
                key = (case_id, model)
                if key in artifacts:
                    raise ValueError(f"{context}: duplicate artifact for {case_id}/{model}")
                relative_code_path = str(Path(relative_case_path) / code_file)
                artifacts[key] = CandidateArtifact(
                    case_id=case_id,
                    model=model,
                    case_path=self.package_root / relative_case_path,
                    code_path=self.package_root / relative_code_path,
                    relative_code_path=relative_code_path,
                    label=_parse_optional_bool(
                        row["paper_metric_passed"],
                        context=f"{context}/paper_metric_passed",
                    ),
                )
                if model not in models:
                    models.append(model)
                if case_id not in case_ids:
                    case_ids.append(case_id)
        if not artifacts:
            raise ValueError(
                f"No manifest rows found for setting {self.setting!r} in {manifest_path}"
            )
        return artifacts, tuple(models), tuple(case_ids)


__all__ = ["CandidateArtifact", "DEFAULT_SETTING", "PrepBenchArtifacts"]
