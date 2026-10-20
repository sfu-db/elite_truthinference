"""Database-held-out evaluation. The only module that reads correctness labels, after predictions are fixed."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import read_questions
from .model import ReliabilityVote


def cross_validate(rows, agreements, models=None, condition_on_empty=True, folds=5, dialect="sqlite", route_budget=0.,
                   structure=True):
    """Fit on the questions of other databases, predict each held-out fold. Reads no labels."""
    questions = read_questions(rows, agreements, models)
    groups = np.array([q.db_id for q in questions])
    predictions, fitted = [None] * len(questions), []
    for fold, (train, test) in enumerate(GroupKFold(folds).split(groups, groups=groups), 1):
        model = ReliabilityVote(condition_on_empty, dialect, route_budget, structure).fit([questions[i] for i in train])
        for i, prediction in zip(test, model.predict([questions[i] for i in test]), strict=True):
            predictions[i] = prediction
        fitted.append({"fold": fold, "test_databases": sorted(set(groups[test])), "model": model.to_dict()})
    return questions, predictions, fitted


def interval(differences):
    d = np.asarray(differences, dtype=float)
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"delta": float(d.mean()), "delta_95ci": [float(d.mean() - half), float(d.mean() + half)]}


def score(rows, questions, predictions):
    sources = sorted({rows[0]["candidates"][p]["source"] for p in questions[0].positions})
    records = []
    for row, q, prediction in zip(rows, questions, predictions, strict=True):
        labels = row["labels"]
        clusters = prediction["clusters"]
        # Labels may come from a different equivalence check than the agreement: a cluster can mix correct
        # and wrong members, and several clusters can hold correct ones. A cluster scores its members' mean.
        correct = [float(np.mean([bool(labels[i]) for i in members])) for members in clusters]
        winners = prediction["decision_clusters"]
        by_source = {row["candidates"][p]["source"]: bool(labels[p]) for p in q.positions}
        records.append({
            "question_id": q.question_id,
            "tf": float(np.mean([correct[k] for k in winners])) if winners else 0.,
            "tf_deterministic": prediction["selected_index"] is not None and bool(labels[prediction["selected_index"]]),
            "oracle": any(correct), "mixed_cluster": any(0 < c < 1 for c in correct), "sources": by_source,
            "confidence": prediction["confidence"], "all_wrong": prediction["probabilities"][0],
            "route": prediction["route"]})
    accuracy = {s: float(np.mean([r["sources"][s] for r in records])) for s in sources}
    best = max(sources, key=accuracy.get)

    mean = lambda key: float(np.mean([r[key] for r in records]))
    tf = np.array([r["tf"] for r in records])
    summary = {"questions": len(records), "tf": mean("tf"), "tf_deterministic": mean("tf_deterministic"),
               "oracle": mean("oracle"), "best_single": float(np.mean([r["sources"][best] for r in records])),
               "vs_best_single": interval(tf - [r["sources"][best] for r in records]),
               "mixed_clusters": sum(r["mixed_cluster"] for r in records)}
    return {"source_accuracy": accuracy, "best_single_source": best, "summary": summary,
            "routing": routing(records), "question_scores": records}


BUDGETS = (.05, .1, .2, .3, .4, .5)


def routing(records):
    """Routing questions to an oracle solver (always correct) under a budget, by different risk rankings.

    `cutoff` evaluates the deployed rule: route when confidence is below the label-free cutoff fitted on
    the training folds. `budgets` routes the given fraction of questions with the lowest score for each
    ranking (ties at the cutoff routed at random, in expectation); random routing is its expectation.
    Accuracy counts a routed question as correct.
    """
    correct = np.array([r["tf"] for r in records])
    rankings = {"confidence": np.array([r["confidence"] for r in records]),
                "not_all_wrong": 1 - np.array([r["all_wrong"] for r in records])}
    wrong = correct < 1
    def routed(score, count, base):
        # Expected accuracy when questions tied at the cutoff are routed at random.
        cutoff = np.sort(score)[count - 1] if count else -np.inf
        below, tied = score < cutoff, score == cutoff
        share = np.where(below, 1., np.where(tied, (count - below.sum()) / max(tied.sum(), 1), 0.))
        return float((base + share * (1 - base)).mean())

    out = {"selection_accuracy": float(correct.mean()),
           "auroc_wrong_selection": {name: float(roc_auc_score(wrong, -score)) if 0 < wrong.sum() < len(wrong) else None
                                     for name, score in rankings.items()},
           "budgets": {}}
    for budget in BUDGETS:
        count = round(budget * len(records))
        row = {"routed": count, "random": float(correct.mean() + budget * (1 - correct).mean())}
        for name, score in rankings.items():
            row[name] = routed(score, count, correct)
        out["budgets"][f"{budget:.2f}"] = row
    routed = np.array([r["route"] for r in records])
    oracle_pool = np.array([r["oracle"] for r in records])
    out["cutoff"] = {"routed": int(routed.sum()), "routed_fraction": float(routed.mean()),
                     "kept_accuracy": float(correct[~routed].mean()) if (~routed).any() else None,
                     "end_to_end_with_oracle": float(np.where(routed, 1., correct).mean())}
    if routed.any():
        out["cutoff"].update(
            # A non-oracle solver must exceed this accuracy on the routed questions to raise overall accuracy.
            break_even_accuracy=float(correct[routed].mean()),
            routed_without_correct_candidate=float((~oracle_pool[routed]).mean()))
    return out


def with_labels(rows, label_rows):
    """Rows scored with another dataset's labels; questions and candidates must match one to one."""
    label_rows = list(label_rows)
    if len(rows) != len(label_rows):
        raise ValueError("Label dataset has a different number of questions")
    out = []
    for row, other in zip(rows, label_rows):
        if (str(row["question_id"]) != str(other["question_id"])
                or [(c["source"], c["text"]) for c in row["candidates"]]
                != [(c["source"], c["text"]) for c in other["candidates"]]):
            raise ValueError(f"Question {row['question_id']}: candidates differ from the label dataset")
        out.append({**row, "labels": other["labels"]})
    return out


def evaluate(rows, agreements, models=None, condition_on_empty=True, folds=5, dialect="sqlite", route_budget=0.,
             structure=True):
    rows = list(rows)
    questions, predictions, fitted = cross_validate(rows, agreements, models, condition_on_empty, folds, dialect,
                                                    route_budget, structure)
    # Every prediction is fixed before any label is read.
    return {**score(rows, questions, predictions), "folds": fitted, "predictions": predictions}
