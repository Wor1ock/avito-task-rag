"""Batch retrieval pipeline entry point.

Automates the full production flow: Feather ingestion (HTML-cleaned articles)
-> hybrid index build/load -> MAP@10 validation on the calibration set ->
ranked top-10 predictions for the test set -> compliant ``answer.csv`` export.

Run from the project root:

    PYTHONUTF8=1 uv run python main.py

Paths can be overridden via the ``RAG_DATA_DIR`` and ``RAG_ANSWER_PATH``
environment variables (used by the mock end-to-end verification).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.dataset import ArticleDataset, load_feather_table
from src.indexer import BM25_FILENAME, FAISS_FILENAME, HybridIndexer
from src.searcher import HybridSearcher
from src.utils import calculate_map_at_10, setup_logger

logger = logging.getLogger("rag.main")

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
ARTICLES_PATH = DATA_DIR / "articles.f"
CALIBRATION_PATH = DATA_DIR / "calibration.f"
TEST_PATH = DATA_DIR / "test.f"
ANSWER_PATH = Path(os.environ.get("RAG_ANSWER_PATH", "answer.csv"))
TOP_K_FINAL = 10


def ensure_index(dataset: ArticleDataset) -> None:
    """Build and persist the hybrid index unless both artifacts already exist.

    Args:
        dataset: Corpus to index when the artifacts are missing.
    """
    bm25_path = DATA_DIR / BM25_FILENAME
    faiss_path = DATA_DIR / FAISS_FILENAME
    if bm25_path.exists() and faiss_path.exists():
        logger.info("Found existing index artifacts (%s, %s): skipping build", bm25_path, faiss_path)
        return
    logger.info("Index artifacts missing: building from %d articles", len(dataset))
    indexer = HybridIndexer()
    indexer.build_index(dataset)
    indexer.save(str(DATA_DIR))


def build_searcher(dataset: ArticleDataset) -> HybridSearcher:
    """Instantiate the searcher, rebuilding stale artifacts on corpus mismatch.

    Args:
        dataset: Corpus the searcher serves.

    Returns:
        Ready-to-query hybrid searcher.
    """
    ensure_index(dataset)
    try:
        return HybridSearcher(dataset, index_dir=str(DATA_DIR))
    except ValueError:
        logger.warning("Index artifacts do not match the current corpus: rebuilding")
        (DATA_DIR / BM25_FILENAME).unlink(missing_ok=True)
        (DATA_DIR / FAISS_FILENAME).unlink(missing_ok=True)
        ensure_index(dataset)
        return HybridSearcher(dataset, index_dir=str(DATA_DIR))


def run_validation(searcher: HybridSearcher) -> None:
    """Compute MAP@10 on the calibration set, if present.

    Args:
        searcher: Ready-to-query hybrid searcher.
    """
    if not CALIBRATION_PATH.exists():
        logger.warning("Calibration file %s not found: skipping validation", CALIBRATION_PATH)
        return
    calibration = load_feather_table(CALIBRATION_PATH, required_columns=("query_id", "query_text", "ground_truth"))
    predictions: list[list[int]] = []
    ground_truths: list[list[int]] = []
    for row in tqdm(calibration.itertuples(index=False), total=len(calibration), desc="calibration"):
        predictions.append(searcher.search(str(row.query_text), top_k_final=TOP_K_FINAL))
        ground_truths.append([int(token) for token in str(row.ground_truth).split()])
    score = calculate_map_at_10(predictions, ground_truths)
    logger.info("Calibration MAP@10 over %d queries: %.4f", len(calibration), score)
    print(f"MAP@10 on calibration ({len(calibration)} queries): {score:.4f}")


def run_test(searcher: HybridSearcher) -> None:
    """Predict ranked top-10 article ids for the test set and export ``answer.csv``.

    Args:
        searcher: Ready-to-query hybrid searcher.
    """
    test = load_feather_table(TEST_PATH, required_columns=("query_id", "query_text"))
    answers: list[str] = []
    for row in tqdm(test.itertuples(index=False), total=len(test), desc="test"):
        ranked = searcher.search(str(row.query_text), top_k_final=TOP_K_FINAL)
        # Defensive dedup (order-preserving); the searcher already returns unique ids.
        unique_ranked = list(dict.fromkeys(ranked))
        answers.append(" ".join(str(article_id) for article_id in unique_ranked))

    submission = pd.DataFrame({"query_id": test["query_id"].astype(int), "answer": answers})
    if len(submission) != len(test):
        raise RuntimeError(f"Submission has {len(submission)} rows for {len(test)} test queries")
    ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(ANSWER_PATH, index=False)
    logger.info("Wrote %d-row submission to %s", len(submission), ANSWER_PATH)
    print(f"Submission written to {ANSWER_PATH} ({len(submission)} rows)")


def main() -> None:
    """Run the batch pipeline: ingest -> index -> validate -> predict -> export."""
    setup_logger(log_file=DATA_DIR / "app.log")

    dataset = ArticleDataset()
    dataset.load_from_feather(ARTICLES_PATH)

    searcher = build_searcher(dataset)
    run_validation(searcher)
    run_test(searcher)


if __name__ == "__main__":
    main()
