"""Fit, predict or evaluate reliability-weighted voting: python -m qrank.tf evaluate --dataset bird-dev:parseval.

The dataset fixes the equivalence check (its agreement databases, from its manifest.json); agreement is the
dataset's stored agreement restricted to the chosen source pool. `--labels` scores against another dataset's
labels, so different equivalence checks can be compared on the same labels.
"""

import argparse
import json
import logging
from pathlib import Path
import time

from datasets import DatasetDict, load_from_disk
import sqlglot

from .agreement import dataset_agreement
from .data import read_questions
from .evaluate import evaluate, with_labels
from .model import ReliabilityVote

ROOT = Path("data")
DATASETS = {
    f"{benchmark}:{check}": ROOT / folder / f"{prefix}_agr_q_sql_af_AQROSN_{code}_m6"
    for benchmark, folder, prefix in (("bird-dev", "bird", "bird_dev"), ("spider-dev", "spider", "spider_dev"),
                                      ("spider-test", "spider", "spider_test"))
    for check, code in {"official": "ss", "parseval": "ss_pv", "testsuite": "ss_ts"}.items()
}
# Spider's empty results are often correct, BIRD's never.
EMPTY_CONDITIONING = {"bird-dev": True, "spider-dev": False, "spider-test": False}


def load(name_or_path, split):
    path = DATASETS[name_or_path] if name_or_path in DATASETS else Path(name_or_path)
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    return path, dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit", "predict", "evaluate"))
    parser.add_argument("--dataset", required=True, help=f"One of {sorted(DATASETS)} or a saved dataset path")
    parser.add_argument("--labels", help="Dataset whose labels score the predictions (evaluate); default --dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", help="Source pool; all sources when omitted")
    parser.add_argument("--empty-conditioning", choices=("on", "off"),
                        help="Default: on for BIRD, off for Spider; required for a dataset path")
    parser.add_argument("--route-budget", type=float, default=0.,
                        help="Fraction of questions to route to a stronger solver, by lowest confidence")
    parser.add_argument("--structure", choices=("on", "off"), default="on",
                        help="SQL structure ranker; turn off for non-SQL candidates (e.g. math answers)")
    parser.add_argument("--dialect", default="sqlite")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", type=Path, help="Saved fit output, required for predict")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, dataset = load(args.dataset, args.split)
    rows = list(dataset)
    manifest = json.loads((path / "manifest.json").read_text())
    started = time.perf_counter()
    if args.command == "predict":
        if args.model is None:
            parser.error("predict requires --model")
        saved = json.loads(args.model.read_text())
        model = ReliabilityVote.from_dict(saved["model"])
        args.models = saved["configuration"]["models"]
        condition_on_empty = model.condition_on_empty
        args.structure = "on" if model.structure else "off"
    elif args.empty_conditioning is not None:
        condition_on_empty = args.empty_conditioning == "on"
    elif args.dataset in DATASETS:
        condition_on_empty = EMPTY_CONDITIONING[args.dataset.split(":")[0]]
    else:
        parser.error("--empty-conditioning is required with a dataset path")
    agreements = dataset_agreement(rows, args.models)
    config = {"dataset": str(path), "split": args.split, "dataset_fingerprint": dataset._fingerprint,
              "databases": manifest["databases"], "models": args.models,
              "condition_on_empty": condition_on_empty, "dialect": args.dialect,
              "route_budget": args.route_budget, "structure": args.structure,
              "sqlglot_version": sqlglot.__version__}
    if args.command == "evaluate":
        label_path, label_dataset = load(args.labels, args.split) if args.labels else (path, dataset)
        scored = with_labels(rows, label_dataset) if args.labels else rows
        result = evaluate(scored, agreements, args.models, condition_on_empty, args.folds, args.dialect,
                          args.route_budget, args.structure == "on")
        config.update(folds=args.folds, labels=str(label_path))
    elif args.command == "fit":
        model = ReliabilityVote(condition_on_empty, args.dialect, args.route_budget, args.structure == "on")
        result = {"model": model.fit(read_questions(rows, agreements, args.models)).to_dict()}
    else:
        result = {"predictions": model.predict(read_questions(rows, agreements, args.models))}
        config.update(dialect=model.dialect, route_budget=model.route_budget,
                      model_path=str(args.model))
    config["elapsed_seconds"] = time.perf_counter() - started
    result["configuration"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.command == "evaluate":
        print(json.dumps({key: result[key] for key in ("best_single_source", "summary", "routing")},
                         indent=2))


if __name__ == "__main__":
    main()
