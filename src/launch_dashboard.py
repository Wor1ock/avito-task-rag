"""Phoenix dashboard for per-query error analysis of calibration retrieval.

Supersedes the plain-text ``src.analysis`` report and the ``src.inspect_article``
CLI: every calibration query becomes a RETRIEVER span in a local Arize Phoenix
instance, carrying the query text, the predicted top-10 articles with the exact
enriched (indexed) text of each document, and an "Error Category" evaluation
that classifies every imperfect query (AP@10 < 1) so spans can be filtered by
error type in the UI, e.g.::

    evals['Error Category'].label == 'Bad Ranking'

Categories (checked in order of severity):

- ``Missing in Index``: a ground-truth article is absent from the indexed corpus.
- ``Bad Retrieval``: a ground-truth article is indexed but not among the fused
  first-stage candidates (top ``top_k_candidates``).
- ``Bad Ranking``: a ground-truth article is retrieved as a candidate but does
  not reach the final top-10.
- ``Semantic Drift``: every ground-truth article is in the top-10, yet AP@10 < 1
  because irrelevant hits are ranked above them (wrong-intent matches score higher).

The final top-10 comes from the persisted calibration predictions written by
``main.py``; the candidate pool is recomputed with the first-stage retrievers
(BM25 + FAISS + RRF, no reranker — reranking only permutes the fused head and
cannot rescue an article that was never retrieved).

Run from the project root (after a ``main.py`` run has produced the
calibration predictions artifact):

    PYTHONUTF8=1 uv run python -m src.launch_dashboard
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Sequence
from collections.abc import Set as AbstractSet

import hydra
import pandas as pd
import phoenix as px
from omegaconf import DictConfig, OmegaConf
from phoenix.trace import DocumentEvaluations, SpanEvaluations, TraceDataset
from tqdm import tqdm

from src.config import AppConfig
from src.dataset import ArticleDataset, load_feather_table
from src.indexer import HybridIndexer
from src.searcher import HybridSearcher
from src.utils import average_precision_at_10, setup_logger

logger = logging.getLogger("rag.launch_dashboard")

CATEGORY_CORRECT = "Correct"
CATEGORY_MISSING_IN_INDEX = "Missing in Index"
CATEGORY_BAD_RETRIEVAL = "Bad Retrieval"
CATEGORY_BAD_RANKING = "Bad Ranking"
CATEGORY_SEMANTIC_DRIFT = "Semantic Drift"

ERROR_CATEGORY_EVAL_NAME = "Error Category"
RELEVANCE_EVAL_NAME = "relevance"
TRACE_DATASET_NAME = "calibration-error-analysis"


def parse_id_string(value: str) -> list[int]:
    """Space-separated id string (submission ``answer`` format) -> int list."""
    return [int(token) for token in str(value).split()]


def categorize_error(
    ap_10: float,
    ground_truth_ids: Sequence[int],
    predicted_ids: Collection[int],
    candidate_ids: Collection[int],
    indexed_ids: AbstractSet[int],
) -> str:
    """Classify a query's retrieval failure mode from pure id-set membership.

    The checks run from the most to the least severe failure, so a query is
    labeled by the worst thing that happened to any of its ground-truth
    articles.

    Args:
        ap_10: AP@10 of the query; values >= 1.0 short-circuit to ``Correct``.
        ground_truth_ids: Relevant article ids for the query.
        predicted_ids: Final ranking submitted for the query (top-10).
        candidate_ids: First-stage fused candidate pool (top ``top_k_candidates``).
        indexed_ids: Every article id present in the indexed corpus.

    Returns:
        One of :data:`CATEGORY_CORRECT`, :data:`CATEGORY_MISSING_IN_INDEX`,
        :data:`CATEGORY_BAD_RETRIEVAL`, :data:`CATEGORY_BAD_RANKING`,
        :data:`CATEGORY_SEMANTIC_DRIFT`.
    """
    if ap_10 >= 1.0:
        return CATEGORY_CORRECT
    truth = set(ground_truth_ids)
    if truth - indexed_ids:
        return CATEGORY_MISSING_IN_INDEX
    if truth - set(candidate_ids):
        return CATEGORY_BAD_RETRIEVAL
    if truth - set(predicted_ids):
        return CATEGORY_BAD_RANKING
    return CATEGORY_SEMANTIC_DRIFT


def explain_category(
    ap_10: float,
    ground_truth_ids: Sequence[int],
    predicted_ids: Collection[int],
    candidate_ids: Collection[int],
    indexed_ids: AbstractSet[int],
) -> str:
    """Human-readable breakdown of where each ground-truth article was lost."""
    predicted = set(predicted_ids)
    candidates = set(candidate_ids)
    not_indexed = [i for i in ground_truth_ids if i not in indexed_ids]
    not_retrieved = [i for i in ground_truth_ids if i in indexed_ids and i not in candidates]
    not_ranked = [i for i in ground_truth_ids if i in candidates and i not in predicted]

    parts = [f"AP@10={ap_10:.3f}"]
    if not_indexed:
        parts.append(f"not in the indexed corpus: {not_indexed}")
    if not_retrieved:
        parts.append(f"indexed but not retrieved as candidates: {not_retrieved}")
    if not_ranked:
        parts.append(f"retrieved but pushed out of the top-10: {not_ranked}")
    if len(parts) == 1:
        if ap_10 >= 1.0:
            parts.append("all ground-truth articles ranked in the top-10 ahead of irrelevant hits")
        else:
            parts.append("all ground-truth articles are in the top-10 but irrelevant hits are ranked above them")
    return "; ".join(parts)


def build_analysis_table(
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    searcher: HybridSearcher,
    indexed_ids: AbstractSet[int],
    top_k_final: int,
    top_k_candidates: int,
) -> pd.DataFrame:
    """Join predictions with the ground truth and categorize every query.

    Rows are joined on ``query_id`` (both sides cast to int), so a predictions
    file produced from a sampled calibration run is analyzed over exactly the
    queries it contains. The fused first-stage candidate pool is recomputed per
    query to separate retrieval misses from ranking misses.

    Args:
        calibration: Ground-truth table with ``query_id``, ``query_text``,
            and ``ground_truth`` (space-separated relevant article ids).
        predictions: Submission-format table with ``query_id`` and ``answer``.
        searcher: First-stage searcher (reranker off) used to rebuild the
            candidate pool.
        indexed_ids: Every article id present in the indexed corpus.
        top_k_final: Rank cutoff of the metric (and of the submitted ranking).
        top_k_candidates: Depth of the recomputed candidate pool.

    Returns:
        One row per analyzed query with columns: ``query_id``, ``query_text``,
        ``ap_10``, ``category``, ``explanation``, ``ground_truth_ids``,
        ``predicted_ids``, ``candidate_scores`` (article_id -> fused score).
    """
    calibration = calibration.assign(query_id=calibration["query_id"].astype(int))
    predictions = predictions.assign(
        query_id=predictions["query_id"].astype(int),
        answer=predictions["answer"].fillna("").astype(str),
    )
    merged = calibration.merge(predictions, on="query_id", how="inner", validate="one_to_one")
    if len(merged) < len(calibration):
        logger.warning(
            "Predictions cover %d of %d calibration queries: analyzing the intersection only",
            len(merged),
            len(calibration),
        )

    rows: list[dict] = []
    for row in tqdm(merged.itertuples(index=False), total=len(merged), desc="candidate retrieval"):
        truth = parse_id_string(row.ground_truth)
        predicted = parse_id_string(row.answer)[:top_k_final]
        candidates = searcher.search_with_scores(
            str(row.query_text), top_k_candidates=top_k_candidates, top_k_final=top_k_candidates
        )
        candidate_ids = [article_id for article_id, _ in candidates]
        ap_10 = average_precision_at_10(predicted, truth, k=top_k_final)
        rows.append(
            {
                "query_id": int(row.query_id),
                "query_text": str(row.query_text),
                "ap_10": ap_10,
                "category": categorize_error(ap_10, truth, predicted, candidate_ids, indexed_ids),
                "explanation": explain_category(ap_10, truth, predicted, candidate_ids, indexed_ids),
                "ground_truth_ids": truth,
                "predicted_ids": predicted,
                "candidate_scores": dict(candidates),
            }
        )
    return pd.DataFrame(rows)


def build_trace_dataset(analysis: pd.DataFrame, dataset: ArticleDataset) -> TraceDataset:
    """Render the analysis table as Phoenix spans with attached evaluations.

    Each query becomes a root RETRIEVER span whose ``retrieval.documents``
    carry the exact enriched (indexed) text of every predicted article, so the
    preprocessed representation can be inspected directly in the UI. Two
    evaluations are attached:

    - span-level :data:`ERROR_CATEGORY_EVAL_NAME`: category label, AP@10 score,
      and a per-query explanation (the ``EvaluationResult`` used for filtering);
    - document-level :data:`RELEVANCE_EVAL_NAME`: relevant/irrelevant flag per
      retrieved document against the ground truth.

    Args:
        analysis: Output of :func:`build_analysis_table`.
        dataset: Loaded article dataset (source of titles and enriched texts).

    Returns:
        TraceDataset ready to be passed to ``phoenix.launch_app``.
    """
    articles_by_id = {
        article.article_id: (article.title, dataset.get_enriched_text(article)) for article in dataset.articles
    }
    base_time = pd.Timestamp.now(tz="UTC")

    span_rows: list[dict] = []
    doc_eval_rows: list[dict] = []
    for position, row in enumerate(analysis.itertuples(index=False)):
        span_id = f"{row.query_id:016x}"
        truth_set = set(row.ground_truth_ids)
        documents = []
        for article_id in row.predicted_ids:
            title, enriched = articles_by_id.get(article_id, ("<unknown>", ""))
            document = {
                "document.id": str(article_id),
                "document.content": enriched,
                "document.metadata": {"title": title, "in_ground_truth": article_id in truth_set},
            }
            score = row.candidate_scores.get(article_id)
            if score is not None:
                document["document.score"] = float(score)
            documents.append(document)

        start_time = base_time + pd.Timedelta(milliseconds=position)
        span_rows.append(
            {
                "name": "hybrid_retrieval",
                "span_kind": "RETRIEVER",
                "parent_id": None,
                "start_time": start_time,
                "end_time": start_time + pd.Timedelta(milliseconds=1),
                "status_code": "OK",
                "status_message": "",
                "context.span_id": span_id,
                "context.trace_id": f"{row.query_id:032x}",
                "attributes.input.value": row.query_text,
                "attributes.output.value": " ".join(str(article_id) for article_id in row.predicted_ids),
                "attributes.retrieval.documents": documents,
            }
        )
        doc_eval_rows.extend(
            {
                "span_id": span_id,
                "position": doc_position,
                "label": "relevant" if article_id in truth_set else "irrelevant",
                "score": float(article_id in truth_set),
            }
            for doc_position, article_id in enumerate(row.predicted_ids)
        )

    error_category_evaluations = SpanEvaluations(
        eval_name=ERROR_CATEGORY_EVAL_NAME,
        dataframe=pd.DataFrame(
            {
                "span_id": [f"{query_id:016x}" for query_id in analysis["query_id"]],
                "label": analysis["category"].to_numpy(),
                "score": analysis["ap_10"].astype(float).to_numpy(),
                "explanation": analysis["explanation"].to_numpy(),
            }
        ).set_index("span_id"),
    )
    relevance_evaluations = DocumentEvaluations(
        eval_name=RELEVANCE_EVAL_NAME,
        dataframe=pd.DataFrame(doc_eval_rows).set_index(["span_id", "position"]),
    )
    return TraceDataset(
        pd.DataFrame(span_rows),
        name=TRACE_DATASET_NAME,
        evaluations=[error_category_evaluations, relevance_evaluations],
    )


def log_summary(analysis: pd.DataFrame) -> None:
    """Log aggregate quality statistics and the error-category distribution."""
    total = len(analysis)
    logger.info("Analyzed %d queries | MAP@10=%.4f", total, float(analysis["ap_10"].mean()))
    for category, count in analysis["category"].value_counts().items():
        logger.info("%-17s: %3d queries (%.1f%%)", category, count, 100 * count / total)


def build_first_stage_searcher(dataset: ArticleDataset, config: AppConfig) -> HybridSearcher:
    """Searcher over the persisted indexes with the reranker forced off.

    The candidate-depth categorization concerns first-stage retrieval only:
    the cross-encoder merely permutes the fused head, and the analyzed top-10
    already comes from the persisted (possibly reranked) predictions.
    """
    encoder = HybridIndexer(
        model_name=config.model.bi_encoder,
        batch_size=config.model.batch_size,
        device=config.model.device,
        max_seq_length=config.model.max_seq_length,
        normalize_embeddings=config.model.normalize_embeddings,
        chunk_size=config.model.chunk_size,
        chunk_overlap=config.model.chunk_overlap,
    )
    return HybridSearcher(
        dataset=dataset,
        encoder=encoder,
        bm25_path=config.path.bm25_index,
        faiss_path=config.path.faiss_index,
        rrf_k=config.hybrid.rrf_k,
        bm25_weight=config.hybrid.bm25_weight,
        dense_weight=config.hybrid.dense_weight,
        aggregation_strategy=config.aggregation.strategy,
        reranker_enabled=False,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Categorize calibration errors and serve them in a Phoenix dashboard.

    Args:
        cfg: Hydra-composed configuration (validated into :class:`AppConfig`).
    """
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    setup_logger(log_file=config.path.data_dir / "app.log")
    # One search per query at full candidate depth: silence the per-query logs.
    logging.getLogger("rag.searcher").setLevel(logging.WARNING)

    if not config.path.calibration_answer.exists():
        raise FileNotFoundError(
            f"Calibration predictions not found at {config.path.calibration_answer}; run main.py first"
        )
    calibration = load_feather_table(
        config.path.calibration, required_columns=("query_id", "query_text", "ground_truth")
    )
    predictions = pd.read_csv(config.path.calibration_answer)
    logger.info("Loaded %d predictions from %s", len(predictions), config.path.calibration_answer)

    dataset = ArticleDataset()
    dataset.load_from_feather(config.path.articles)
    searcher = build_first_stage_searcher(dataset, config)
    indexed_ids = frozenset(article.article_id for article in dataset.articles)

    analysis = build_analysis_table(
        calibration,
        predictions,
        searcher,
        indexed_ids,
        top_k_final=config.top_k_final,
        top_k_candidates=config.top_k_candidates,
    )
    log_summary(analysis)

    session = px.launch_app(trace=build_trace_dataset(analysis, dataset))
    logger.info(
        "Phoenix dashboard at %s — filter spans with evals['%s'].label; press Ctrl+C to stop",
        session.url,
        ERROR_CATEGORY_EVAL_NAME,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down the Phoenix dashboard")
        px.close_app()


if __name__ == "__main__":
    main()
