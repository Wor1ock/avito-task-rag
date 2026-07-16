"""Multi-stage hybrid retrieval: BM25 + FAISS candidates re-ranked by a cross-encoder.

Pipeline per query:

    1. Lexical: tokenize the query and take the top BM25 candidates.
    2. Semantic: encode/normalize the query and search the FAISS index.
    3. Fusion: union the two candidate sets (deduplicated corpus indices).
    4. Reranking: score (query, enriched document) pairs with a cross-encoder.
    5. Final ranking: sort by cross-encoder score, return top article_ids.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from src.dataset import ArticleDataset
from src.indexer import BM25_FILENAME, DEFAULT_EMBEDDING_MODEL, FAISS_FILENAME, HybridIndexer
from src.utils import tokenize

logger = logging.getLogger("rag.searcher")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class HybridSearcher:
    """Combines lexical and semantic retrieval with cross-encoder re-ranking."""

    def __init__(
        self,
        dataset: ArticleDataset,
        index_dir: str = "data/",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ) -> None:
        """
        Args:
            dataset: Loaded article dataset (source of candidate texts; must be
                the same corpus, in the same order, the indexes were built on).
            index_dir: Directory holding the saved BM25/FAISS artifacts.
            model_name: Bi-encoder checkpoint (must match the one used at build time).
            reranker_name: Cross-encoder checkpoint for re-ranking.
            device: Torch device for both models; auto-detected when None.
        """
        self.dataset = dataset
        self.index_dir = Path(index_dir)
        self.reranker_name = reranker_name
        self.device = device
        # Reuse the indexer's lazy encoder so query vectors get the same
        # normalization as the corpus vectors.
        self._encoder = HybridIndexer(model_name=model_name, device=device)
        self.bm25 = None
        self.faiss_index: faiss.Index | None = None
        self.article_ids: list[int] = []
        self._enriched_corpus: list[str] = []
        self.reranker: CrossEncoder | None = None
        self.load_index()

    def load_index(self) -> None:
        """Load persisted artifacts and initialize both models.

        Raises:
            FileNotFoundError: If the BM25 or FAISS artifact is missing.
            ValueError: If the dataset's article order does not match the
                ``article_ids`` mapping the indexes were built with.
        """
        bm25_path = self.index_dir / BM25_FILENAME
        faiss_path = self.index_dir / FAISS_FILENAME
        with bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.article_ids = payload["article_ids"]
        self.faiss_index = faiss.read_index(str(faiss_path))
        logger.info(
            "Loaded BM25 (%d docs) and FAISS (%d vectors) from %s",
            self.bm25.corpus_size,
            self.faiss_index.ntotal,
            self.index_dir,
        )

        dataset_ids = [article.article_id for article in self.dataset.articles]
        if dataset_ids != self.article_ids:
            raise ValueError("Dataset article order does not match the saved index mapping; rebuild the index")
        self._enriched_corpus = self.dataset.get_enriched_corpus()

        start = time.perf_counter()
        self.reranker = CrossEncoder(self.reranker_name, device=self.device)
        logger.info("Loaded reranker %s in %.1fs", self.reranker_name, time.perf_counter() - start)

    def _search_lexical(self, query: str, top_k: int) -> list[int]:
        """Top ``top_k`` corpus indices by BM25 score (decreasing)."""
        scores = self.bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(scores))
        return np.argsort(scores)[::-1][:top_k].tolist()

    def _search_semantic(self, query: str, top_k: int) -> list[int]:
        """Top ``top_k`` corpus indices by cosine similarity (decreasing)."""
        query_vec = self._encoder.encode([query])
        top_k = min(top_k, self.faiss_index.ntotal)
        _, indices = self.faiss_index.search(query_vec, top_k)
        return [int(i) for i in indices[0] if i != -1]

    def search(self, query: str, top_k_candidates: int = 30, top_k_final: int = 10) -> list[int]:
        """Run the full multi-stage pipeline for a single query.

        Args:
            query: Raw query text.
            top_k_candidates: Candidates fetched per first-stage retriever.
            top_k_final: Number of article ids in the final ranking.

        Returns:
            ``article_id`` list ordered by decreasing cross-encoder relevance,
            length <= ``top_k_final``.
        """
        start = time.perf_counter()
        lexical = self._search_lexical(query, top_k_candidates)
        semantic = self._search_semantic(query, top_k_candidates)

        # Sorted for a deterministic reranker input order.
        candidates = sorted(set(lexical) | set(semantic))
        pairs = [[query, self._enriched_corpus[idx]] for idx in candidates]
        ce_scores = self.reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates, ce_scores, strict=True), key=lambda item: item[1], reverse=True)
        result = [self.article_ids[idx] for idx, _ in ranked[:top_k_final]]
        logger.info(
            "Query served in %.2fs (%d lexical + %d semantic -> %d unique candidates)",
            time.perf_counter() - start,
            len(lexical),
            len(semantic),
            len(candidates),
        )
        return result
