"""Label-free question records: sources, validity, emptiness and execution clusters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    question_id: str
    db_id: str
    # Original candidate indices of the pool, in input order.
    positions: tuple[int, ...]
    # Original indices of the valid pool candidates; the model only sees these.
    valid: tuple[int, ...]
    sources: tuple[str, ...]
    nonempty: tuple[bool, ...]
    # Execution clusters as indices into `valid`, ordered by first member.
    clusters: tuple[tuple[int, ...], ...]
    # SQL and unary query features of every pool candidate, aligned with `positions` (cluster ranker).
    sql: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]


def pool_positions(row, models):
    """Candidate indices of the requested sources; every source must appear exactly once."""
    if models is None:
        return list(range(len(row["candidates"])))
    positions = [i for i, c in enumerate(row["candidates"]) if c["source"] in models]
    if sorted(row["candidates"][i]["source"] for i in positions) != sorted(models):
        raise ValueError(f"Question {row['question_id']} lacks exactly one candidate per requested source")
    return positions


def execution_clusters(agreement, valid):
    """Clusters of an agreement matrix, an equivalence relation on `valid`."""
    clusters = []
    for a in valid:
        members = tuple(b for b in valid if agreement[a][b])
        if members not in clusters:
            clusters.append(members)
    return clusters


def read_questions(rows, agreements, models=None):
    """One record per row; correctness labels and gold SQL are never read.

    `agreements[k]` is row k's agreement over the pool's candidates, in pool order
    (see agreement.py for how it is built).
    """
    questions = []
    for row, agreement in zip(rows, agreements, strict=True):
        positions = pool_positions(row, models)
        candidates = [row["candidates"][i] for i in positions]
        valid = tuple(i for i in positions if not row["candidates"][i]["invalid_mask"])
        local = {original: k for k, original in enumerate(valid)}
        in_pool = {original: k for k, original in enumerate(positions)}
        clusters = execution_clusters({a: {b: agreement[in_pool[a]][in_pool[b]] for b in valid} for a in valid}, valid)
        questions.append(Question(
            str(row["question_id"]), row["metadata"]["db_id"], tuple(positions), valid,
            tuple(row["candidates"][i]["source"] for i in valid),
            tuple(row["candidates"][i]["is_non_empty_exec"] for i in valid),
            tuple(tuple(local[i] for i in c) for c in clusters),
            tuple(c["text"] for c in candidates), tuple(tuple(c["features"]) for c in candidates)))
    return questions
