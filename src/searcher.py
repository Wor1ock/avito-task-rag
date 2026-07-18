"""Multi-stage hybrid retrieval: BM25 + FAISS fused via weighted Reciprocal Rank Fusion.

Pipeline per query:

    1. Lexical: tokenize the query and take the top BM25 candidates.
    2. Semantic: encode the raw query, search the chunk-level FAISS index, and
       deduplicate chunk hits to their parent articles (best chunk wins).
    3. Fusion: weighted RRF over both article rankings
       (``score = sum(weight / (rrf_k + rank))``).
    4. Optional reranking: when enabled, score (query, enriched document)
       pairs with a cross-encoder and re-sort the fused candidates.
    5. Final ranking: top article_ids by fused (or reranker) score.
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
from src.indexer import HybridIndexer
from src.utils import tokenize

logger = logging.getLogger("rag.searcher")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class HybridSearcher:
    """Combines lexical and semantic retrieval with weighted RRF fusion.

    A cross-encoder re-ranking stage over the fused candidates is applied only
    when ``reranker_enabled`` is set.
    """

    def __init__(
        self,
        dataset: ArticleDataset,
        encoder: HybridIndexer,
        bm25_path: str | Path,
        faiss_path: str | Path,
        *,
        rrf_k: float,
        bm25_weight: float,
        dense_weight: float,
        reranker_enabled: bool = False,
        reranker_name: str | None = None,
        rerank_depth: int = 15,
        device: str | None = None,
    ) -> None:
        """
        Args:
            dataset: Loaded article dataset (source of candidate texts; must be
                the same corpus, in the same order, the indexes were built on).
            encoder: Indexer whose bi-encoder encodes queries; must be configured
                identically to the one that built the FAISS index.
            bm25_path: Persisted BM25 artifact (pickled index + article_ids).
            faiss_path: Persisted FAISS index.
            rrf_k: Reciprocal Rank Fusion constant (``hybrid.rrf_k``).
            bm25_weight: RRF weight of the lexical ranking (``hybrid.bm25_weight``).
            dense_weight: RRF weight of the semantic ranking (``hybrid.dense_weight``).
            reranker_enabled: Whether to re-rank fused candidates with a
                cross-encoder.
            reranker_name: Cross-encoder checkpoint; defaults to
                :data:`DEFAULT_RERANKER_MODEL` when None and reranking is enabled.
            rerank_depth: How many top fused candidates the cross-encoder
                re-scores (``reranker.rerank_depth``); the RRF tail keeps its
                fusion order.
            device: Torch device for the reranker; auto-detected when None.
        """
        self.dataset = dataset
        self.bm25_path = Path(bm25_path)
        self.faiss_path = Path(faiss_path)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.reranker_enabled = reranker_enabled
        self.reranker_name = reranker_name or DEFAULT_RERANKER_MODEL
        self.rerank_depth = rerank_depth
        self.device = device
        self._encoder = encoder
        self.bm25 = None
        self.faiss_index: faiss.Index | None = None
        self.article_ids: list[int] = []
        self.chunk_article_ids: list[int] = []
        # FAISS chunk position -> position of the parent article in article_ids.
        self._chunk_parent_positions: list[int] = []
        self._enriched_corpus: list[str] = []
        self.reranker: CrossEncoder | None = None
        self.load_index()

    def load_index(self) -> None:
        """Load persisted artifacts and, when enabled, the reranker model.

        Raises:
            FileNotFoundError: If the BM25 or FAISS artifact is missing.
            ValueError: If the dataset's article order or chunking does not
                match the mappings the indexes were built with (stale
                artifacts are rebuilt by the caller).
        """
        with self.bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.article_ids = payload["article_ids"]
        if "chunk_article_ids" not in payload:
            raise ValueError("BM25 artifact predates chunk-level indexing; rebuild the index")
        self.chunk_article_ids = payload["chunk_article_ids"]
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_path}")
        self.faiss_index = faiss.read_index(str(self.faiss_path))
        logger.info(
            "Loaded BM25 (%d docs) from %s and FAISS (%d chunk vectors) from %s",
            self.bm25.corpus_size,
            self.bm25_path,
            self.faiss_index.ntotal,
            self.faiss_path,
        )

        dataset_ids = [article.article_id for article in self.dataset.articles]
        if dataset_ids != self.article_ids:
            raise ValueError("Dataset article order does not match the saved index mapping; rebuild the index")
        if self.faiss_index.ntotal != len(self.chunk_article_ids):
            raise ValueError("FAISS index size does not match the saved chunk mapping; rebuild the index")
        _, derived_parents = self.dataset.get_chunked_corpus(self._encoder.chunk_size, self._encoder.chunk_overlap)
        if derived_parents != self.chunk_article_ids:
            raise ValueError("Dataset chunking does not match the saved chunk mapping; rebuild the index")
        article_positions = {article_id: pos for pos, article_id in enumerate(self.article_ids)}
        self._chunk_parent_positions = [article_positions[article_id] for article_id in self.chunk_article_ids]

        if self.reranker_enabled:
            # The enriched corpus is only consumed by the rerank stage; skip
            # rendering it entirely when reranking is disabled.
            self._enriched_corpus = self.dataset.get_enriched_corpus()
            start = time.perf_counter()
            # max_length caps the (query, document) pair at 512 tokens: without
            # it the tokenizer pads/attends up to the model maximum (8192 for
            # bge-reranker-v2-m3), which is prohibitively slow on CPU.
            self.reranker = CrossEncoder(self.reranker_name, device=self.device, max_length=512)
            logger.info("Loaded reranker %s in %.1fs", self.reranker_name, time.perf_counter() - start)
        else:
            logger.info("Reranker disabled: final ranking uses RRF fusion scores")

    def _search_lexical(self, query: str, top_k: int) -> list[int]:
        """Top ``top_k`` corpus indices by BM25 score (decreasing)."""
        scores = self.bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(scores))
        return np.argsort(scores)[::-1][:top_k].tolist()

    def _search_semantic(self, query: str, top_k: int) -> list[int]:
        """Top ``top_k`` article corpus positions by best-chunk similarity (decreasing).

        FAISS ranks chunks; hits are deduplicated to their parent article,
        which keeps the rank of its best chunk. The fetch depth doubles until
        ``top_k`` distinct articles are covered or the index is exhausted.
        """
        query_vec = self._encoder.encode([query])
        ntotal = self.faiss_index.ntotal
        fetch = min(top_k * 4, ntotal)
        while True:
            _, indices = self.faiss_index.search(query_vec, fetch)
            positions: list[int] = []
            seen: set[int] = set()
            for chunk_idx in indices[0]:
                if chunk_idx == -1:
                    continue
                position = self._chunk_parent_positions[chunk_idx]
                if position not in seen:
                    seen.add(position)
                    positions.append(position)
            if len(positions) >= top_k or fetch >= ntotal:
                return positions[:top_k]
            fetch = min(fetch * 2, ntotal)

    def _fuse(self, rankings: list[tuple[float, list[int]]]) -> list[tuple[int, float]]:
        """Weighted Reciprocal Rank Fusion over ranked corpus-index lists.

        Args:
            rankings: ``(weight, ranked_indices)`` pairs, best-first rankings.

        Returns:
            ``(corpus_index, fused_score)`` pairs sorted by decreasing score
            (ties broken by corpus index for determinism).
        """
        fused: dict[int, float] = {}
        for weight, ranking in rankings:
            for rank, corpus_idx in enumerate(ranking, start=1):
                fused[corpus_idx] = fused.get(corpus_idx, 0.0) + weight / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def search(self, query: str, top_k_candidates: int = 100, top_k_final: int = 10) -> list[int]:
        """Run the full multi-stage pipeline for a single query.

        Args:
            query: Raw query text.
            top_k_candidates: Candidates fetched per first-stage retriever.
            top_k_final: Number of article ids in the final ranking.

        Returns:
            ``article_id`` list ordered by decreasing relevance,
            length <= ``top_k_final``.
        """
        return [article_id for article_id, _ in self.search_with_scores(query, top_k_candidates, top_k_final)]

    def search_with_scores(
        self,
        query: str,
        top_k_candidates: int = 100,
        top_k_final: int = 10,
    ) -> list[tuple[int, float]]:
        """Like :meth:`search`, but keeps the relevance scores.

        Args:
            query: Raw query text.
            top_k_candidates: Candidates fetched per first-stage retriever.
            top_k_final: Number of article ids in the final ranking.

        Returns:
            ``(article_id, score)`` pairs ordered by decreasing score, where the
            score is the cross-encoder relevance when reranking is enabled and
            the fused RRF score otherwise. Length <= ``top_k_final``.
        """
        start = time.perf_counter()
        lexical = self._search_lexical(query, top_k_candidates)
        semantic = self._search_semantic(query, top_k_candidates)
        candidates = self._fuse([(self.bm25_weight, lexical), (self.dense_weight, semantic)])

        if self.reranker is not None:
            rerank_depth = min(self.rerank_depth, len(candidates))
            indices = [corpus_idx for corpus_idx, _ in candidates[:rerank_depth]]
            remaining_candidates = candidates[rerank_depth:]

            pairs = [[query, self._enriched_corpus[corpus_idx]] for corpus_idx in indices]
            ce_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)

            reranked_candidates = sorted(
                zip(indices, (float(score) for score in ce_scores), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )

            # List order alone decides the final slice (no re-sort below), so the
            # RRF tail simply backfills after the reranked head when
            # top_k_final exceeds rerank_depth.
            candidates = reranked_candidates + remaining_candidates

        result = [(self.article_ids[corpus_idx], float(score)) for corpus_idx, score in candidates[:top_k_final]]
        logger.info(
            "Query served in %.2fs (%d lexical + %d semantic -> %d fused candidates, reranker %s)",
            time.perf_counter() - start,
            len(lexical),
            len(semantic),
            len(candidates),
            "on" if self.reranker is not None else "off",
        )
        return result
