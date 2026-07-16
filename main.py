"""Batch retrieval pipeline entry point.

Automates the full production flow: Feather ingestion (HTML-cleaned articles)
-> hybrid index build/load -> MAP@10 validation on the calibration set ->
ranked top-10 predictions for the test set -> compliant ``answer.csv`` export.

Every parameter comes from the Hydra config tree under ``configs/`` (composed
root: ``configs/config.yaml``), validated into :class:`src.config.AppConfig`.
Run from the project root; any value can be overridden on the command line:

    PYTHONUTF8=1 uv run python main.py
    PYTHONUTF8=1 uv run python main.py path.data_dir=mock_data path.submission=mock_answer.csv
"""

from __future__ import annotations

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.config import AppConfig
from src.dataset import ArticleDataset, load_feather_table
from src.indexer import HybridIndexer
from src.searcher import HybridSearcher
from src.utils import calculate_map_at_10, set_seed, setup_logger

logger = logging.getLogger("rag.main")


def build_encoder(config: AppConfig) -> HybridIndexer:
    """Instantiate the bi-encoder/indexer from the model config.

    Args:
        config: Validated application config.

    Returns:
        Indexer used both for corpus index builds and query encoding.
    """
    return HybridIndexer(
        model_name=config.model.bi_encoder,
        batch_size=config.model.batch_size,
        device=config.model.device,
        max_seq_length=config.model.max_seq_length,
        normalize_embeddings=config.model.normalize_embeddings,
    )


def ensure_index(dataset: ArticleDataset, encoder: HybridIndexer, config: AppConfig) -> None:
    """Build and persist the hybrid index unless both artifacts already exist.

    Args:
        dataset: Corpus to index when the artifacts are missing.
        encoder: Configured indexer performing the build.
        config: Validated application config (artifact paths).
    """
    if config.path.bm25_index.exists() and config.path.faiss_index.exists():
        logger.info(
            "Found existing index artifacts (%s, %s): skipping build",
            config.path.bm25_index,
            config.path.faiss_index,
        )
        return
    logger.info("Index artifacts missing: building from %d articles", len(dataset))
    encoder.build_index(dataset)
    encoder.save(config.path.bm25_index, config.path.faiss_index)


def build_searcher(dataset: ArticleDataset, config: AppConfig) -> HybridSearcher:
    """Instantiate the searcher, rebuilding stale artifacts on corpus mismatch.

    Args:
        dataset: Corpus the searcher serves.
        config: Validated application config.

    Returns:
        Ready-to-query hybrid searcher.
    """
    encoder = build_encoder(config)
    searcher_kwargs = {
        "dataset": dataset,
        "encoder": encoder,
        "bm25_path": config.path.bm25_index,
        "faiss_path": config.path.faiss_index,
        "rrf_k": config.hybrid.rrf_k,
        "bm25_weight": config.hybrid.bm25_weight,
        "dense_weight": config.hybrid.dense_weight,
        "reranker_enabled": config.reranker.enabled,
        "reranker_name": config.reranker.model_name,
        "device": config.model.device,
    }
    ensure_index(dataset, encoder, config)
    try:
        return HybridSearcher(**searcher_kwargs)
    except ValueError:
        logger.warning("Index artifacts do not match the current corpus: rebuilding")
        config.path.bm25_index.unlink(missing_ok=True)
        config.path.faiss_index.unlink(missing_ok=True)
        ensure_index(dataset, encoder, config)
        return HybridSearcher(**searcher_kwargs)


def run_validation(searcher: HybridSearcher, config: AppConfig) -> None:
    """Compute MAP@10 on the calibration set, if present.

    Args:
        searcher: Ready-to-query hybrid searcher.
        config: Validated application config (calibration path, top_k settings).
    """
    if not config.path.calibration.exists():
        logger.warning("Calibration file %s not found: skipping validation", config.path.calibration)
        return
    calibration = load_feather_table(
        config.path.calibration, required_columns=("query_id", "query_text", "ground_truth")
    )
    predictions: list[list[int]] = []
    ground_truths: list[list[int]] = []
    for row in tqdm(calibration.itertuples(index=False), total=len(calibration), desc="calibration"):
        ranked = searcher.search(
            str(row.query_text), top_k_candidates=config.top_k_candidates, top_k_final=config.top_k_final
        )
        predictions.append(ranked)
        ground_truths.append([int(token) for token in str(row.ground_truth).split()])
    score = calculate_map_at_10(predictions, ground_truths)
    logger.info("Calibration MAP@10 over %d queries: %.4f", len(calibration), score)
    print(f"MAP@10 on calibration ({len(calibration)} queries): {score:.4f}")


def run_test(searcher: HybridSearcher, config: AppConfig) -> None:
    """Predict ranked top-k article ids for the test set and export the submission.

    Args:
        searcher: Ready-to-query hybrid searcher.
        config: Validated application config (test path, submission path, top_k settings).
    """
    test = load_feather_table(config.path.test, required_columns=("query_id", "query_text"))
    answers: list[str] = []
    for row in tqdm(test.itertuples(index=False), total=len(test), desc="test"):
        ranked = searcher.search(
            str(row.query_text), top_k_candidates=config.top_k_candidates, top_k_final=config.top_k_final
        )
        # Defensive dedup (order-preserving); the searcher already returns unique ids.
        unique_ranked = list(dict.fromkeys(ranked))
        answers.append(" ".join(str(article_id) for article_id in unique_ranked))

    submission = pd.DataFrame({"query_id": test["query_id"].astype(int), "answer": answers})
    if len(submission) != len(test):
        raise RuntimeError(f"Submission has {len(submission)} rows for {len(test)} test queries")
    config.path.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.path.submission, index=False)
    logger.info("Wrote %d-row submission to %s", len(submission), config.path.submission)
    print(f"Submission written to {config.path.submission} ({len(submission)} rows)")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the batch pipeline: ingest -> index -> validate -> predict -> export.

    Args:
        cfg: Hydra-composed configuration (validated into :class:`AppConfig`).
    """
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    set_seed(config.seed)
    setup_logger(log_file=config.path.data_dir / "app.log")
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    dataset = ArticleDataset()
    dataset.load_from_feather(config.path.articles)

    searcher = build_searcher(dataset, config)
    run_validation(searcher, config)
    run_test(searcher, config)


if __name__ == "__main__":
    main()
