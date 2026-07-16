"""Data loading, HTML cleaning, and text chunking.

This module turns raw article/query tables (Feather / Parquet) into a flat
chunk-level corpus ready for indexing:

    load_table -> clean_html -> chunk_text -> build_chunk_corpus
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Chunk:
    """A single indexable unit of text.

    Attributes:
        chunk_id: Globally unique chunk identifier.
        doc_id: Identifier of the source article/document.
        text: Cleaned chunk text.
    """

    chunk_id: int
    doc_id: int
    text: str


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a dataset table from Feather or Parquet.

    The format is inferred from the file extension (``.f`` / ``.feather``
    are read as Feather, ``.parquet`` as Parquet).

    Args:
        path: Path to the dataset file.

    Returns:
        The loaded table.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is not supported.
    """
    raise NotImplementedError


def clean_html(raw_html: str) -> str:
    """Strip HTML markup and normalize whitespace.

    Uses :class:`~bs4.BeautifulSoup` to extract visible text, dropping
    script/style contents and collapsing consecutive whitespace.

    Args:
        raw_html: Raw HTML string (may also be plain text).

    Returns:
        Cleaned plain-text string.
    """
    raise NotImplementedError


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a text into overlapping character-based chunks.

    Args:
        text: Cleaned input text.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunk strings (empty list for empty input).

    Raises:
        ValueError: If ``chunk_overlap >= chunk_size``.
    """
    raise NotImplementedError


def build_chunk_corpus(
    articles: pd.DataFrame,
    chunk_size: int,
    chunk_overlap: int,
    text_column: str = "text",
    id_column: str = "doc_id",
) -> list[Chunk]:
    """Build the chunk-level corpus from an articles table.

    For every article: clean the HTML, split it into chunks, and assign
    globally unique chunk ids while keeping the mapping back to ``doc_id``.

    Args:
        articles: Table with at least ``id_column`` and ``text_column``.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        text_column: Name of the column holding raw article HTML/text.
        id_column: Name of the column holding the document identifier.

    Returns:
        Flat list of :class:`Chunk` objects covering the whole corpus.
    """
    raise NotImplementedError


def save_chunk_metadata(chunks: list[Chunk], path: str | Path) -> None:
    """Persist the chunk -> document mapping as a Parquet file.

    Args:
        chunks: Chunk corpus produced by :func:`build_chunk_corpus`.
        path: Destination Parquet path (parent directories are created).
    """
    raise NotImplementedError
