"""Batch prediction pipeline: query table -> submission-format predictions table.

Single shared step for both calibration validation and test-set inference, so
local MAP@10 is computed over the exact table layout that is submitted.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src.searcher import HybridSearcher

logger = logging.getLogger("rag.predict")


def predict(
    df: pd.DataFrame,
    searcher: HybridSearcher,
    top_k: int = 10,
    top_k_candidates: int = 100,
    desc: str = "predict",
) -> pd.DataFrame:
    """Rank the top-``top_k`` articles for every query, in submission format.

    Args:
        df: Query table with ``query_id`` and ``query_text`` columns
            (calibration or test set).
        searcher: Ready-to-query hybrid searcher.
        top_k: Number of article ids per query.
        top_k_candidates: Candidates fetched per first-stage retriever.
        desc: Progress-bar label.

    Returns:
        DataFrame with the exact submission layout: ``query_id`` (int) and
        ``answer`` — a space-separated string of order-preserving,
        deduplicated top-``top_k`` article ids. One row per input query,
        in input order.

    Raises:
        ValueError: If a required column is missing from ``df``.
    """
    missing = [column for column in ("query_id", "query_text") if column not in df.columns]
    if missing:
        raise ValueError(f"Query dataframe is missing required columns {missing}; found {list(df.columns)}")

    answers: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc):
        ranked = searcher.search(str(row.query_text), top_k_candidates=top_k_candidates, top_k_final=top_k)
        # Defensive dedup (order-preserving); the searcher already returns unique ids.
        unique_ranked = list(dict.fromkeys(ranked))[:top_k]
        answers.append(" ".join(str(article_id) for article_id in unique_ranked))

    predictions = pd.DataFrame({"query_id": df["query_id"].astype(int).to_numpy(), "answer": answers})
    logger.info("Predicted top-%d rankings for %d queries", top_k, len(predictions))
    return predictions
