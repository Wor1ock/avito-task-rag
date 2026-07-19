"""многостадийный гибридный поиск: BM25 + FAISS со взвешенным Reciprocal Rank Fusion.

пайплайн на запрос:

    1. лексика: токенизация запроса и топ кандидатов BM25.
    2. семантика: кодирование сырого запроса, поиск по чанковому индексу FAISS
       и агрегация сходств чанков к родительским статьям выбранной стратегией
       (max_p / avg_p / sum_p).
    3. слияние: взвешенный RRF по обоим ранжированиям
       (``score = sum(weight / (rrf_k + rank))``).
    4. опциональный реранкинг: кросс-энкодер оценивает пары (сырой запрос,
       лучший чанк статьи) и пересортировывает слитых кандидатов.
    5. финал: топ article_id по слитому (или реранкерному) скору.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from src.dataset import ArticleDataset
from src.indexer import HybridIndexer
from src.utils import tokenize

logger = logging.getLogger("rag.searcher")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# стратегии агрегации скоров чанков в скор статьи для плотной ветки:
# скор статьи выводится из сходств её найденных чанков
CHUNK_AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "max_p": max,
    "avg_p": lambda scores: sum(scores) / len(scores),
    "sum_p": sum,
}


class HybridSearcher:
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
        aggregation_strategy: str = "max_p",
        reranker_enabled: bool = False,
        reranker_name: str | None = None,
        rerank_depth: int = 15,
        device: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.bm25_path = Path(bm25_path)
        self.faiss_path = Path(faiss_path)
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        if aggregation_strategy not in CHUNK_AGGREGATORS:
            raise ValueError(
                f"Unknown aggregation strategy {aggregation_strategy!r}; expected one of {sorted(CHUNK_AGGREGATORS)}"
            )
        self.aggregation_strategy = aggregation_strategy
        self.reranker_enabled = reranker_enabled
        self.reranker_name = reranker_name or DEFAULT_RERANKER_MODEL
        self.rerank_depth = rerank_depth
        self.device = device
        self._encoder = encoder
        self.bm25 = None
        self.faiss_index: faiss.Index | None = None
        self.article_ids: list[int] = []
        self.chunk_article_ids: list[int] = []
        # позиция чанка в FAISS -> позиция родительской статьи в article_ids
        self._chunk_parent_positions: list[int] = []
        # позиция родительской статьи -> позиции её чанков в FAISS
        self._article_chunk_indices: dict[int, list[int]] = {}
        self._chunk_corpus: list[str] = []
        self.reranker: CrossEncoder | None = None
        self.load_index()

    def load_index(self) -> None:
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
            "загружены BM25 (%d документов) из %s и FAISS (%d векторов чанков) из %s",
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
        chunks, derived_parents = self.dataset.get_chunked_corpus(self._encoder.chunk_size, self._encoder.chunk_overlap)
        if derived_parents != self.chunk_article_ids:
            raise ValueError("Dataset chunking does not match the saved chunk mapping; rebuild the index")
        self._chunk_corpus = chunks
        article_positions = {article_id: pos for pos, article_id in enumerate(self.article_ids)}
        self._chunk_parent_positions = [article_positions[article_id] for article_id in self.chunk_article_ids]
        self._article_chunk_indices = {}
        for chunk_idx, position in enumerate(self._chunk_parent_positions):
            self._article_chunk_indices.setdefault(position, []).append(chunk_idx)

        if self.reranker_enabled:
            start = time.perf_counter()
            # max_length ограничивает пару (запрос, документ) 512 токенами: без
            # него токенизатор паддит до максимума модели (8192 у
            # bge-reranker-v2-m3), что неприемлемо медленно на CPU
            self.reranker = CrossEncoder(self.reranker_name, device=self.device, max_length=512)
            logger.info("реранкер %s загружен за %.1f с", self.reranker_name, time.perf_counter() - start)
        else:
            logger.info("реранкер отключён: финальное ранжирование по скорам слияния RRF")

    def _search_lexical(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(scores))
        return np.argsort(scores)[::-1][:top_k].tolist()

    def _search_semantic(self, query_vec: np.ndarray, top_k: int) -> list[int]:
        ntotal = self.faiss_index.ntotal
        aggregate = CHUNK_AGGREGATORS[self.aggregation_strategy]
        fetch = min(top_k * 4, ntotal)
        while True:
            scores, indices = self.faiss_index.search(query_vec, fetch)
            chunk_scores: dict[int, list[float]] = {}
            for score, chunk_idx in zip(scores[0], indices[0], strict=True):
                if chunk_idx == -1:
                    continue
                chunk_scores.setdefault(self._chunk_parent_positions[chunk_idx], []).append(float(score))
            if len(chunk_scores) >= top_k or fetch >= ntotal:
                ranked = sorted(
                    chunk_scores.items(),
                    key=lambda item: (-aggregate(item[1]), item[0]),
                )
                return [position for position, _ in ranked[:top_k]]
            fetch = min(fetch * 2, ntotal)

    def _best_chunk_index(self, position: int, query_vec: np.ndarray) -> int:
        chunk_indices = self._article_chunk_indices[position]
        if len(chunk_indices) == 1:
            return chunk_indices[0]
        vectors = np.stack([self.faiss_index.reconstruct(chunk_idx) for chunk_idx in chunk_indices])
        return chunk_indices[int(np.argmax(vectors @ query_vec[0]))]

    def _fuse(self, rankings: list[tuple[float, list[int]]]) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for weight, ranking in rankings:
            for rank, corpus_idx in enumerate(ranking, start=1):
                fused[corpus_idx] = fused.get(corpus_idx, 0.0) + weight / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def search(self, query: str, top_k_candidates: int = 100, top_k_final: int = 10) -> list[int]:
        return [article_id for article_id, _ in self.search_with_scores(query, top_k_candidates, top_k_final)]

    def search_with_scores(
        self,
        query: str,
        top_k_candidates: int = 100,
        top_k_final: int = 10,
    ) -> list[tuple[int, float]]:
        start = time.perf_counter()
        # запрос нормализуется только в ветке BM25 (tokenize() внутри
        # _search_lexical убирает пунктуацию и стоп-слова и лемматизирует);
        # би-энкодер и реранкер получают сырую строку запроса
        lexical = self._search_lexical(query, top_k_candidates)
        query_vec = self._encoder.encode([query])
        semantic = self._search_semantic(query_vec, top_k_candidates)
        candidates = self._fuse([(self.bm25_weight, lexical), (self.dense_weight, semantic)])

        if self.reranker is not None:
            rerank_depth = min(self.rerank_depth, len(candidates))
            indices = [corpus_idx for corpus_idx, _ in candidates[:rerank_depth]]
            remaining_candidates = candidates[rerank_depth:]

            # каждая статья-кандидат представлена своим лучшим чанком, поэтому
            # кросс-энкодер оценивает пассаж, который вероятнее всего отвечает
            # на запрос, а не усечённый полный документ
            pairs = [
                [query, self._chunk_corpus[self._best_chunk_index(corpus_idx, query_vec)]] for corpus_idx in indices
            ]
            ce_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)

            reranked_candidates = sorted(
                zip(indices, (float(score) for score in ce_scores), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )

            # финальный срез определяется только порядком списка (ниже пересортировки
            # нет), поэтому хвост RRF просто добирается после реранкнутой головы,
            # когда top_k_final больше rerank_depth
            candidates = reranked_candidates + remaining_candidates

        result = [(self.article_ids[corpus_idx], float(score)) for corpus_idx, score in candidates[:top_k_final]]
        logger.info(
            "запрос обработан за %.2f с (%d лексических + %d семантических -> %d слитых кандидатов, реранкер %s)",
            time.perf_counter() - start,
            len(lexical),
            len(semantic),
            len(candidates),
            "вкл" if self.reranker is not None else "выкл",
        )
        return result
