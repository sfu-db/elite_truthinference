"""SQL-structure features of execution clusters, used by the cluster ranker."""

from functools import lru_cache

import numpy as np
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

COMPONENTS = ("tables", "joins", "where", "projection", "aggregates", "group_by", "order_limit", "distinct")


@lru_cache(maxsize=None)
def components(sql, dialect="sqlite"):
    """Canonical components of one query, or None when it is not exactly one parsable query.

    Table aliases are replaced by table names (per query, not per scope); quoting and
    identifier case are normalized; WHERE conjuncts and set-valued components are sorted.
    A component is None when the query does not have it. Cached; callers must not mutate the result.
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None
    tree = normalize_identifiers(statements[0], dialect=dialect)
    aliases = {t.alias: t.name for t in tree.find_all(exp.Table) if t.alias}

    def canon(node):
        node = node.copy()
        for column in node.find_all(exp.Column):
            if column.table in aliases:
                column.set("table", exp.to_identifier(aliases[column.table]))
        for table in node.find_all(exp.Table):
            table.set("alias", None)
        node = node.transform(lambda n: n.this if isinstance(n, exp.Alias) else n)
        return node.sql(dialect=dialect, identify=False)

    def conjuncts(node):
        node = node.unnest()
        return conjuncts(node.left) + conjuncts(node.right) if isinstance(node, exp.And) else [node]

    def sorted_or_none(values):
        values = sorted(values)
        return tuple(values) if values else None

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    where = tree.args.get("where")
    order, limit = tree.args.get("order"), tree.args.get("limit")
    return {
        "tables": sorted_or_none(t.name for t in tree.find_all(exp.Table)),
        "joins": sorted_or_none(canon(j.args["on"]) for j in tree.find_all(exp.Join) if j.args.get("on")),
        "where": sorted_or_none(canon(c) for c in conjuncts(where.this)) if where else None,
        "projection": tuple(canon(e) for e in tree.selects),
        "aggregates": sorted_or_none(canon(a) for a in tree.find_all(exp.AggFunc)),
        "group_by": canon(tree.args["group"]) if tree.args.get("group") else None,
        "order_limit": (canon(order) if order else None, canon(limit) if limit else None) if order or limit else None,
        "distinct": bool(select is not None and select.args.get("distinct")),
    }


def cluster_features(question, dialect="sqlite"):
    """One row per execution cluster: the members' mean log1p unary query features, then the cluster's
    structural match to the other clusters per component and overall.

    The match for a component is the share of (member, other-cluster member) pairs of parsable queries
    whose component is equal; 0 when there is no such pair. Invalid candidates are ignored.
    """
    by_position = dict(zip(question.positions, question.sql))
    unary = dict(zip(question.positions, question.features))
    clusters = [[question.valid[i] for i in c] for c in question.clusters]
    parsed = {p: components(by_position[p], dialect) for c in clusters for p in c}
    rows = []
    for k, members in enumerate(clusters):
        others = [p for u, c in enumerate(clusters) if u != k for p in c if parsed[p]]
        match = []
        for r in COMPONENTS:
            values = [parsed[a][r] == parsed[b][r] for a in members if parsed[a] for b in others]
            match.append(np.mean(values) if values else 0.)
        mean_unary = np.log1p(np.maximum(np.array([unary[p] for p in members], dtype=float), 0)).mean(axis=0)
        rows.append(np.r_[mean_unary, match, np.mean(match)])
    return np.array(rows)
