from __future__ import annotations
from dataclasses import dataclass, fields

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import traverse_scope

_PREDICATE_TYPES = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.Like,
    exp.ILike,
    exp.In,
    exp.Between,
    exp.Is,
    exp.RegexpLike,
    exp.Glob,
)

_AGG_FUNCS = (
    exp.Sum,
    exp.Avg,
    exp.Min,
    exp.Max,
    exp.Count,
    exp.StddevPop,
    exp.StddevSamp,
    exp.VariancePop,
    exp.Variance,
    exp.GroupConcat,
    exp.ArrayAgg,
)

_EMPTY_KWARGS = dict(
    num_distinct=0, num_joins=0, num_filters=0,
    num_agg_funcs=0, num_subquery=0,
    num_having=0, num_union=0, num_tables=0,
    num_columns=0, offset=0, limit=0,
    max_nesting_depth=0, num_cte=0, is_correlated_subquery=0,
    has_self_join=0, num_window_funcs=0, num_case_when=0,
    num_literals=0, num_group_by_cols=0, num_order_by_cols=0,
    has_ambiguous_column=0, set_op_mixed=0, num_projections=0,
    has_star_projection=0, top_k=0
)


@dataclass(frozen=True)
class QueryFeatures:
    num_distinct: int
    num_joins: int
    num_filters: int
    num_agg_funcs: int
    num_subquery: int
    num_having: int
    num_union: int
    num_tables: int
    num_columns: int
    offset: int  # concrete OFFSET value, 0 if none; outermost only
    limit: int  # concrete LIMIT value, 0 if none; outermost only
    max_nesting_depth: int  # deepest subquery nesting level, 0 if none
    num_cte: int  # number of WITH-clause CTEs
    is_correlated_subquery: int  # 1 if any subquery references an outer scope column
    has_self_join: int  # 1 if the same table is joined to itself under different aliases
    num_window_funcs: int  # number of window expressions
    num_case_when: int  # count of CASE WHEN expressions
    num_literals: int  # count of string/numeric constant literals (excludes LIMIT/OFFSET values)
    num_group_by_cols: int  # total grouping columns across all GROUP BY clauses (0 => no GROUP BY)
    num_order_by_cols: int  # total ordering columns across all query-level ORDER BY clauses (0 => none; excludes window-internal ORDER BY)
    has_ambiguous_column: int  # 1 if an unqualified column appears with >1 table in scope
    set_op_mixed: int  # 1 if more than one distinct set-op type is used (UNION + EXCEPT, etc.)
    num_projections: int  # number of items in the outermost SELECT list (output column count)
    has_star_projection: int  # 1 if the outermost SELECT list includes * or alias.*
    top_k: int = 0  # 1 if outermost has LIMIT + ORDER BY together (and no OFFSET)

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def values(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in self.feature_names())

    @classmethod
    def from_sql(cls, sql: str, dialect="sqlite"):
        try:
            expr = parse_one(sql, read=dialect)
        except Exception:
            expr = None
        if expr is None:
            return cls(**_EMPTY_KWARGS)

        all_selects = list(expr.find_all(exp.Select))
        num_distinct = sum(1 for s in all_selects if s.args.get("distinct"))
        num_joins = len(list(expr.find_all(exp.Join)))
        num_having = len(list(expr.find_all(exp.Having)))
        num_agg_funcs = len(list(expr.find_all(_AGG_FUNCS)))

        def _set_op_kind(node):
            kind = node.args.get("kind")
            return str(kind).upper() if kind else type(node).__name__.upper()

        set_op_nodes = list(expr.find_all(exp.SetOperation))
        num_union = len(set_op_nodes)
        set_op_mixed = 1 if len({_set_op_kind(n) for n in set_op_nodes}) > 1 else 0

        order_nodes = [s.args["order"] for s in all_selects if s.args.get("order")]
        num_order_by_cols = sum(len(o.expressions) for o in order_nodes)

        group_nodes = [s.args["group"] for s in all_selects if s.args.get("group")]
        num_group_by_cols = sum(len(g.expressions) for g in group_nodes)

        num_filters = sum(
            len(list(where.find_all(_PREDICATE_TYPES)))
            for where in expr.find_all(exp.Where)
        )

        top_level_selects = (
            set(id(s) for s in all_selects if s is expr)
            if isinstance(expr, exp.Select)
            else set(
                id(s) for s in expr.find_all(exp.Select)
                if s.parent is expr or isinstance(s.parent, exp.SetOperation)
            )
        )
        num_subquery = sum(1 for s in all_selects if id(s) not in top_level_selects)

        def _select_depth(select_node):
            depth = 0
            node = select_node.parent
            while node:
                if isinstance(node, exp.Select):
                    depth += 1
                node = node.parent
            return depth

        max_nesting_depth = max((_select_depth(s) for s in all_selects), default=0)

        num_tables = len({t.sql(dialect=dialect) for t in expr.find_all(exp.Table)})
        num_columns = len({
            c.sql(dialect=dialect) for c in expr.find_all(exp.Column)
            if not isinstance(c.this, exp.Star)
        })

        num_cte = len(list(expr.find_all(exp.CTE)))
        num_window_funcs = len(list(expr.find_all(exp.Window)))
        num_case_when = len(list(expr.find_all(exp.Case)))
        num_literals = sum(
            1 for l in expr.find_all(exp.Literal)
            if l.find_ancestor(exp.Limit, exp.Offset) is None
        )

        has_self_join = 0
        for s in all_selects:
            names = [t.name for t in s.find_all(exp.Table) if t.name]
            if len(names) != len(set(names)):
                has_self_join = 1
                break

        is_correlated_subquery = 0
        has_ambiguous_column = 0
        try:
            for scope in traverse_scope(expr):
                local_sources = set(scope.selected_sources.keys())
                if not scope.is_root:
                    for col in scope.columns:
                        table_ref = col.text("table")
                        if table_ref and table_ref not in local_sources:
                            is_correlated_subquery = 1
                            break

                if len(local_sources) > 1 and scope.unqualified_columns:
                    has_ambiguous_column = 1
        except Exception:
            pass

        order = expr.args.get("order")
        limit_expr = expr.args.get("limit")
        offset_expr = expr.args.get("offset")

        def _int_value(node):
            if node is None:
                return 0
            target = node.expression if hasattr(node, "expression") else node
            try:
                return int(target.this)
            except (TypeError, ValueError, AttributeError):
                return 0

        limit = _int_value(limit_expr)
        offset = _int_value(offset_expr)

        top_k = 1 if (limit > 0 and order is not None and offset == 0) else 0

        projection_root = expr.this if isinstance(expr, exp.With) else expr
        while isinstance(projection_root, exp.SetOperation):
            projection_root = projection_root.this

        num_projections = 0
        has_star_projection = 0
        if isinstance(projection_root, exp.Select):
            proj_exprs = projection_root.expressions
            num_projections = len(proj_exprs)
            has_star_projection = 1 if any(
                isinstance(e, exp.Star) or
                (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))
                for e in proj_exprs
            ) else 0

        return cls(
            num_distinct=num_distinct,
            num_joins=num_joins,
            num_filters=num_filters,
            num_agg_funcs=num_agg_funcs,
            num_subquery=num_subquery,
            num_having=num_having,
            num_union=num_union,
            num_tables=num_tables,
            num_columns=num_columns,
            offset=offset,
            limit=limit,
            max_nesting_depth=max_nesting_depth,
            num_cte=num_cte,
            is_correlated_subquery=is_correlated_subquery,
            has_self_join=has_self_join,
            num_window_funcs=num_window_funcs,
            num_case_when=num_case_when,
            num_literals=num_literals,
            num_group_by_cols=num_group_by_cols,
            num_order_by_cols=num_order_by_cols,
            has_ambiguous_column=has_ambiguous_column,
            set_op_mixed=set_op_mixed,
            num_projections=num_projections,
            has_star_projection=has_star_projection,
            top_k=top_k,
        )


        
# @dataclass(frozen=True)
# class QueryFeatures:
#     num_select: int
#     num_distinct: int
#     num_joins: int
#     num_filters: int
#     num_agg_funcs: int
#     num_subquery: int
#     num_group_by: int
#     num_order_by: int
#     num_having: int
#     num_union: int
#     num_tables: int
#     num_columns: int
#     asc_or_desc: int  # 0, 1, -1, 1 ASC, 0 DESC, -1 if no ORDER BY, we only consider the outmost ORDER BY
#     offset: int  # Offset clause, 0 or concrete value, 0 if no OFFSET, we only consider the outmost ORDER BY
#     limit: int  # Limit caluse, 0 or concrete value, 0 if no LIMIT, we only consider the outmost ORDER BY
#     return_null: int = 0  # whether the query returns null value, 0 if no, 1 if yes
#     syntax_error: int = 0  # whether the query has syntax error, 0 if no, 1 if yes
#     top_k: int = 0  # whether the query has top-k, 0 if no, 1 if yes


#     @classmethod
#     def from_sql(cls, sql: str, dialect="sqlite"):
#         expr = parse_one(sql, read=dialect)
        
#         sql_lower = sql.lower()
#         asc_or_desc, limit, offset = 0, 0, 0
#         num_tables, num_columns = 0, 0
#         num_predicates, num_agg_funcs = 0, 0
#         num_select = sql_lower.count("select ")
#         num_distinct = sql_lower.count("distinct ")
#         num_joins = sql_lower.count(" join ")
#         num_order_by = sql_lower.count(" order by ")
#         num_having = sql_lower.count(" having ")
#         num_union = (
#             sql_lower.count(" union ")
#             + sql_lower.count(" intersect ")
#             + sql_lower.count(" except ")
#         )
        
