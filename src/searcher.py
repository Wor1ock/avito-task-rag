"""Hybrid retrieval: BM25 + dense FAISS candidates fused with Reciprocal Rank Fusion.

Pipeline per query:

    1. Retrieve ``top_k_candidates`` chunk ids from BM25 and from FAISS.
    2. Fuse the two ranked lists with RRF (per-retriever weights).
    3. Aggregate chunk scores to document level (max over a document's chunks).
    4. Optionally re-rank the fused candidates with a cross-encoder.
    5. Return the ``top_k_final`` document ids.
"""

from __future__ import annotations

from src.indexer import BM25Indexer, FaissIndexer


class HybridSearcher:
    """Combines a sparse and a dense retriever with weighted RRF fusion."""

    def __init__(
        self,
        bm25_indexer: BM25Indexer,
        faiss_indexer: FaissIndexer,
        chunk_to_doc: dict[int, int],
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        """
        Args:
            bm25_indexer: Fitted sparse retriever.
            faiss_indexer: Fitted dense retriever.
            chunk_to_doc: Mapping from chunk id to source document id.
            rrf_k: RRF smoothing constant (rank offset).
            bm25_weight: Weight of the BM25 ranked list in the fusion.
            dense_weight: Weight of the dense ranked list in the fusion.
        """
        self.bm25_indexer = bm25_indexer
        self.faiss_indexer = faiss_indexer
        self.chunk_to_doc = chunk_to_doc
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def _search_bm25(self, query: str, top_k: int) -> list[int]:
        """Return the top ``top_k`` chunk ids by BM25 score.

        Args:
            query: Raw query text.
            top_k: Number of candidates to return.

        Returns:
            Chunk ids ordered by decreasing BM25 score.
        """
        raise NotImplementedError

    def _search_dense(self, query: str, top_k: int) -> list[int]:
        """Return the top ``top_k`` chunk ids by embedding similarity.

        Args:
            query: Raw query text.
            top_k: Number of candidates to return.

        Returns:
            Chunk ids ordered by decreasing inner-product similarity.
        """
        raise NotImplementedError

    def _rrf_fuse(self, ranked_lists: list[tuple[list[int], float]]) -> dict[int, float]:
        """Fuse ranked candidate lists with weighted Reciprocal Rank Fusion.

        For each item at (0-based) rank ``r`` in a list with weight ``w``,
        its contribution is ``w / (rrf_k + r + 1)``; contributions are summed
        across lists.

        Args:
            ranked_lists: Pairs of (ranked chunk ids, list weight).

        Returns:
            Mapping from chunk id to fused RRF score.
        """
        raise NotImplementedError

    def _aggregate_to_documents(self, chunk_scores: dict[int, float]) -> dict[int, float]:
        """Collapse chunk-level scores to document level.

        A document's score is the maximum score among its chunks.

        Args:
            chunk_scores: Fused chunk id -> score mapping.

        Returns:
            Document id -> score mapping.
        """
        raise NotImplementedError

    def rerank(self, query: str, doc_ids: list[int], top_k: int) -> list[int]:
        """Re-rank candidate documents with a cross-encoder (placeholder).

        The baseline returns the input order truncated to ``top_k``; a real
        implementation should score (query, document) pairs with a
        cross-encoder and sort by that score.

        Args:
            query: Raw query text.
            doc_ids: Candidate document ids from the fusion stage.
            top_k: Number of documents to keep.

        Returns:
            Re-ranked document ids, length <= ``top_k``.
        """
        raise NotImplementedError

    def search(self, query: str, top_k_candidates: int = 100, top_k_final: int = 10) -> list[int]:
        """Run the full hybrid retrieval pipeline for a single query.

        Args:
            query: Raw query text.
            top_k_candidates: Candidates per first-stage retriever.
            top_k_final: Documents in the final ranked answer.

        Returns:
            Ranked document ids, length <= ``top_k_final``.
        """
        raise NotImplementedError

    def search_batch(
        self,
        queries: list[str],
        top_k_candidates: int = 100,
        top_k_final: int = 10,
    ) -> list[list[int]]:
        """Vectorized/batched variant of :meth:`search` for many queries.

        Args:
            queries: Raw query texts.
            top_k_candidates: Candidates per first-stage retriever.
            top_k_final: Documents in each final ranked answer.

        Returns:
            One ranked document id list per query, in input order.
        """
        raise NotImplementedError
