"""построение гибридного индекса: BM25 по статьям и плотный FAISS по чанкам.

пайплайн: ArticleDataset -> (обогащённые тексты -> токенизированный корпус ->
BM25Okapi, чанки с заголовком -> эмбеддинги -> L2-нормализация -> IndexFlatIP)
-> сохранённые артефакты, каждая позиция FAISS привязана к article_id родительской статьи.
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


class HybridIndexer:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        device: str | None = None,
        max_seq_length: int | None = None,
        normalize_embeddings: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model: SentenceTransformer | None = None
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.article_ids: list[int] = []
        self.chunk_article_ids: list[int] = []

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            start = time.perf_counter()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length
            logger.info("модель эмбеддингов %s загружена за %.1f с", self.model_name, time.perf_counter() - start)
        return self.model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, dataset: ArticleDataset) -> None:
        if len(dataset) == 0:
            raise ValueError("Cannot build an index over an empty dataset")
        self.article_ids = [article.article_id for article in dataset.articles]

        start = time.perf_counter()
        tokenized_corpus = dataset.get_tokenized_corpus()
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("индекс BM25 по %d документам построен за %.2f с", len(dataset), time.perf_counter() - start)

        start = time.perf_counter()
        chunks, self.chunk_article_ids = dataset.get_chunked_corpus(self.chunk_size, self.chunk_overlap)
        embeddings = self.encode(chunks, show_progress=True)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        logger.info(
            "индекс FAISS IndexFlatIP (%d векторов чанков по %d статьям, dim=%d) построен за %.2f с",
            self.faiss_index.ntotal,
            len(dataset),
            embeddings.shape[1],
            time.perf_counter() - start,
        )

    def save(self, bm25_path: str | Path, faiss_path: str | Path) -> None:
        if self.bm25 is None or self.faiss_index is None:
            raise RuntimeError("Nothing to save: call build_index() first")
        bm25_path = Path(bm25_path)
        faiss_path = Path(faiss_path)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)

        with bm25_path.open("wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "article_ids": self.article_ids, "chunk_article_ids": self.chunk_article_ids},
                f,
            )
        faiss.write_index(self.faiss_index, str(faiss_path))
        logger.info("индекс BM25 сохранён в %s, индекс FAISS — в %s", bm25_path, faiss_path)
