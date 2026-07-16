"""Hybrid index construction: sparse BM25 and dense FAISS over enriched articles.

Pipeline: ArticleDataset -> enriched texts -> (tokenized corpus -> BM25Okapi,
dense embeddings -> L2-normalize -> IndexFlatIP) -> persisted artifacts.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.dataset import ArticleDataset

logger = logging.getLogger("rag.indexer")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BM25_FILENAME = "bm25.pkl"
FAISS_FILENAME = "faiss.index"


class HybridIndexer:
    """Builds and persists a sparse (BM25) and a dense (FAISS) index in lockstep.

    Both indexes are built over the same enriched corpus in the same order, so
    a position in either index maps to the same entry of :attr:`article_ids`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: sentence-transformers bi-encoder checkpoint.
            batch_size: Encoding batch size.
            device: Torch device ("cpu"/"cuda"); auto-detected when None.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model: SentenceTransformer | None = None
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.article_ids: list[int] = []

    def _load_model(self) -> SentenceTransformer:
        """Lazily instantiate the bi-encoder (kept for query encoding reuse)."""
        if self.model is None:
            start = time.perf_counter()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Loaded embedding model %s in %.1fs", self.model_name, time.perf_counter() - start)
        return self.model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts into an L2-normalized float32 matrix (rows: cosine-ready).

        Args:
            texts: Texts to embed (corpus or queries).
            show_progress: Whether to display an encoding progress bar.

        Returns:
            Array of shape ``(len(texts), dim)`` with unit-norm rows, so inner
            product in :class:`faiss.IndexFlatIP` equals cosine similarity.
        """
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, dataset: ArticleDataset) -> None:
        """Build both indexes from the dataset's enriched corpus.

        Args:
            dataset: Loaded article dataset.

        Raises:
            ValueError: If the dataset is empty.
        """
        if len(dataset) == 0:
            raise ValueError("Cannot build an index over an empty dataset")
        self.article_ids = [article.article_id for article in dataset.articles]

        start = time.perf_counter()
        tokenized_corpus = dataset.get_tokenized_corpus()
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents in %.2fs", len(dataset), time.perf_counter() - start)

        start = time.perf_counter()
        embeddings = self.encode(dataset.get_enriched_corpus(), show_progress=True)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        logger.info(
            "Built FAISS IndexFlatIP (%d vectors, dim=%d) in %.2fs",
            self.faiss_index.ntotal,
            embeddings.shape[1],
            time.perf_counter() - start,
        )

    def save(self, output_dir: str = "data/") -> None:
        """Persist both indexes and the article id mapping.

        Writes ``bm25.pkl`` (pickled BM25 + article_ids) and ``faiss.index``
        into ``output_dir``.

        Args:
            output_dir: Destination directory (created if missing).

        Raises:
            RuntimeError: If :meth:`build_index` has not been called.
        """
        if self.bm25 is None or self.faiss_index is None:
            raise RuntimeError("Nothing to save: call build_index() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        bm25_path = out / BM25_FILENAME
        with bm25_path.open("wb") as f:
            pickle.dump({"bm25": self.bm25, "article_ids": self.article_ids}, f)
        faiss_path = out / FAISS_FILENAME
        faiss.write_index(self.faiss_index, str(faiss_path))
        logger.info("Saved BM25 index to %s and FAISS index to %s", bm25_path, faiss_path)
