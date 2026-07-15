"""Evaluation metrics and small shared helpers."""

from __future__ import annotations


def average_precision_at_k(relevant: set[int], predicted: list[int], k: int = 10) -> float:
    """Average Precision at ``k`` for a single query.

    AP@k = (1 / min(|relevant|, k)) * sum_{i=1..k} P(i) * rel(i), where
    ``P(i)`` is precision at cutoff ``i`` and ``rel(i)`` indicates whether
    the item at rank ``i`` is relevant.

    Args:
        relevant: Ground-truth relevant document ids.
        predicted: Ranked predicted document ids (best first).
        k: Rank cutoff.

    Returns:
        AP@k in [0, 1]; 0.0 when ``relevant`` is empty.
    """
    raise NotImplementedError


def map_at_k(
    ground_truth: dict[int, set[int]],
    predictions: dict[int, list[int]],
    k: int = 10,
) -> float:
    """Mean Average Precision at ``k`` over a query set (MAP@10 by default).

    Args:
        ground_truth: Query id -> set of relevant document ids.
        predictions: Query id -> ranked predicted document ids.
        k: Rank cutoff.

    Returns:
        Mean of AP@k over all queries present in ``ground_truth``; queries
        missing from ``predictions`` contribute 0.
    """
    raise NotImplementedError


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Seed value.
    """
    raise NotImplementedError
