"""Baseline pipeline entry point.

Orchestrates: data loading -> preprocessing/chunking -> index building ->
hybrid search on the calibration set -> MAP@10 evaluation -> test submission stub.

Run with Hydra overrides, e.g.:

    uv run python main.py model.device=cpu top_k_final=10
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import build_chunk_corpus, load_table, save_chunk_metadata
from src.indexer import BM25Indexer, FaissIndexer
from src.searcher import HybridSearcher
from src.utils import map_at_k, set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the full retrieval baseline described by the Hydra config.

    Args:
        cfg: Composed configuration (see ``configs/config.yaml``).
    """
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # --- 1. Load data -----------------------------------------------------
    articles = load_table(cfg.path.articles)
    calibration = load_table(cfg.path.calibration)
    test = load_table(cfg.path.test)
    log.info(
        "Loaded %d articles, %d calibration queries, %d test queries",
        len(articles), len(calibration), len(test),
    )

    # --- 2. Preprocess: clean HTML and chunk ------------------------------
    chunks = build_chunk_corpus(
        articles,
        chunk_size=cfg.model.chunk_size,
        chunk_overlap=cfg.model.chunk_overlap,
    )
    save_chunk_metadata(chunks, cfg.path.chunk_metadata)
    log.info("Built %d chunks", len(chunks))

    # --- 3. Build indexes --------------------------------------------------
    bm25 = BM25Indexer()
    bm25.build(chunks)
    bm25.save(cfg.path.bm25_index)

    dense = FaissIndexer(
        model_name=cfg.model.bi_encoder,
        device=cfg.model.device,
        batch_size=cfg.model.batch_size,
        max_seq_length=cfg.model.max_seq_length,
        normalize_embeddings=cfg.model.normalize_embeddings,
    )
    dense.build(chunks)
    dense.save(cfg.path.faiss_index)

    # --- 4. Hybrid search on the calibration set ---------------------------
    searcher = HybridSearcher(
        bm25_indexer=bm25,
        faiss_indexer=dense,
        chunk_to_doc={c.chunk_id: c.doc_id for c in chunks},
        rrf_k=cfg.hybrid.rrf_k,
        bm25_weight=cfg.hybrid.bm25_weight,
        dense_weight=cfg.hybrid.dense_weight,
    )
    calibration_predictions = {
        # TODO: adapt column names ("query_id", "query") to the real schema.
        row["query_id"]: searcher.search(
            row["query"],
            top_k_candidates=cfg.top_k_candidates,
            top_k_final=cfg.top_k_final,
        )
        for _, row in calibration.iterrows()
    }

    # --- 5. Evaluate MAP@10 -------------------------------------------------
    # TODO: build ground_truth from the calibration table's relevance column.
    ground_truth: dict[int, set[int]] = {}
    score = map_at_k(ground_truth, calibration_predictions, k=cfg.top_k_final)
    log.info("Calibration MAP@%d = %.4f", cfg.top_k_final, score)

    # --- 6. Generate test submission stub ------------------------------------
    # TODO: run searcher over the test queries and write cfg.path.submission
    # in the competition's expected format (query_id, ranked doc ids).
    log.info("Submission stub would be written to %s", cfg.path.submission)


if __name__ == "__main__":
    main()
