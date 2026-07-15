"""Index construction: sparse BM25 and dense FAISS indexes over the chunk corpus."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.dataset import Chunk


class BM25Indexer:
    """Builds, persists, and loads a BM25 index over tokenized chunks."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[int] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 (lowercasing + simple word splitting).

        Args:
            text: Input string.

        Returns:
            List of tokens.
        """
        raise NotImplementedError

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenize the corpus and fit :class:`~rank_bm25.BM25Okapi`.

        Args:
            chunks: Chunk corpus to index.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Serialize the fitted index (and chunk id order) to disk.

        Args:
            path: Destination pickle path.

        Raises:
            RuntimeError: If :meth:`build` has not been called.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        """Load a previously saved BM25 index.

        Args:
            path: Path produced by :meth:`save`.

        Returns:
            Restored indexer instance.
        """
        raise NotImplementedError


class FaissIndexer:
    """Encodes chunks with a bi-encoder and stores embeddings in a FAISS index."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunk_ids: list[int] = []

    def _load_model(self) -> SentenceTransformer:
        """Lazily instantiate the sentence-transformers bi-encoder.

        Returns:
            The loaded model, moved to :attr:`device`.
        """
        raise NotImplementedError

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts into a float32 embedding matrix.

        Args:
            texts: Texts to embed (chunks or queries).
            show_progress: Whether to display an encoding progress bar.

        Returns:
            Array of shape ``(len(texts), dim)``; L2-normalized when
            :attr:`normalize_embeddings` is set (so inner product == cosine).
        """
        raise NotImplementedError

    def build(self, chunks: list[Chunk]) -> None:
        """Encode the corpus and populate a flat inner-product FAISS index.

        Args:
            chunks: Chunk corpus to index.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write the FAISS index (and chunk id order) to disk.

        Args:
            path: Destination path for the ``.faiss`` file.

        Raises:
            RuntimeError: If :meth:`build` has not been called.
        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str,
        device: str = "cuda",
        **encode_kwargs: object,
    ) -> "FaissIndexer":
        """Load a previously saved FAISS index.

        Args:
            path: Path produced by :meth:`save`.
            model_name: Bi-encoder used at build time (needed to encode queries).
            device: Device for query encoding.
            **encode_kwargs: Extra encoding parameters (batch size, etc.).

        Returns:
            Restored indexer instance.
        """
        raise NotImplementedError
