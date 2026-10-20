"""Execution agreement of a source pool: the dataset's stored agreement restricted to the pool.

The dataset builder compares results as sets on the original database and, for ParSEval datasets,
on every generated test database of the question.
"""

from .data import pool_positions


def dataset_agreement(rows, models):
    """The stored agreement restricted to the pool, over pool positions in input order."""
    agreements = []
    for row in rows:
        positions = pool_positions(row, models)
        agreements.append([[row["agreement"][a][b] for b in positions] for a in positions])
    return agreements
