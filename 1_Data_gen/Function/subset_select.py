"""Deterministic, class-diverse subset selection.

`stratified_subset(df, n, by_cols)` picks `n` rows from `df` such that:

  * Same input `(df, n, by_cols)` always returns the exact same rows in the
    exact same order — no RNG involved.
  * The selection is class-balanced: rows are drawn round-robin from groups
    keyed by `by_cols`, so a small N still spans all (or most) categories.
  * The final order is by `sort_col` (typically the original parquet row
    position).

Used by:
  * MMStar   — stratified_subset(by_cols=["l2_category"])
  * OCRBench — stratified_subset(by_cols=["question_type"])
"""

from __future__ import annotations

from typing import Iterable, List, Optional


def stratified_subset(df, n: int, by_cols: Optional[Iterable[str]] = None,
                      sort_col: str = "index"):
    """Return a deterministic class-balanced subset of `df`."""
    if n is None or n <= 0:
        raise ValueError("stratified_subset: n must be a positive integer.")
    if sort_col not in df.columns:
        raise ValueError(
            f"stratified_subset: sort_col={sort_col!r} not in df columns "
            f"({list(df.columns)})."
        )

    by_cols = list(by_cols) if by_cols else []
    for c in by_cols:
        if c not in df.columns:
            raise ValueError(
                f"stratified_subset: stratification column {c!r} missing."
            )

    total = len(df)
    if n >= total:
        return df.sort_values(sort_col, kind="stable").reset_index(drop=True)

    df_sorted = df.sort_values(sort_col, kind="stable").reset_index(drop=True)

    if not by_cols:
        step = max(1, total // n)
        positions = list(range(0, total, step))[:n]
        if len(positions) < n:
            extra = [i for i in range(total) if i not in set(positions)]
            positions += extra[: n - len(positions)]
        positions.sort()
        return df_sorted.iloc[positions].reset_index(drop=True)

    pos_by_key: dict = {}
    grouped = df_sorted.groupby(list(by_cols), sort=True)
    for key, sub in grouped:
        pos_by_key[key] = list(sub.index)

    keys: List = list(pos_by_key.keys())
    pointers = {k: 0 for k in keys}
    chosen: List[int] = []
    while len(chosen) < n:
        progressed = False
        for k in keys:
            if len(chosen) >= n:
                break
            p = pointers[k]
            bucket = pos_by_key[k]
            if p < len(bucket):
                chosen.append(bucket[p])
                pointers[k] = p + 1
                progressed = True
        if not progressed:
            break

    chosen.sort()
    return df_sorted.iloc[chosen].reset_index(drop=True)
