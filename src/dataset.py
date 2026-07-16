"""Data management layer: article loading, validation, and text enrichment.

Pipeline: JSON articles -> pydantic validation -> enriched text (title
boosting) -> normalized token corpus for lexical (BM25) and semantic indexing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from src.utils import tokenize

logger = logging.getLogger("rag.dataset")

# The title is repeated this many times at the start of the enriched text so
# title words carry more weight in both BM25 term frequencies and embeddings.
TITLE_BOOST_REPEATS = 3


class Article(BaseModel):
    """A single help-center article."""

    article_id: int
    title: str
    text: str


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


class ArticleDataset:
    """In-memory article store with validation and enrichment helpers."""

    def __init__(self) -> None:
        self.articles: list[Article] = []

    def __len__(self) -> int:
        return len(self.articles)

    def load_from_json(self, file_path: str) -> None:
        """Load and validate articles from a JSON file.

        Args:
            file_path: Path to a JSON file containing a list of article objects
                with ``article_id``, ``title``, and ``text`` fields.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If the top-level JSON value is not a list.
            pydantic.ValidationError: If any article fails schema validation.
        """
        path = Path(file_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON list of articles, got {type(raw).__name__}")
        self.articles = [Article.model_validate(item) for item in raw]
        logger.info("Loaded %d articles from %s", len(self.articles), path)

    def get_enriched_text(self, article: Article) -> str:
        """Concatenate title and body, repeating the title to boost its weight.

        Args:
            article: Validated article.

        Returns:
            ``title`` repeated :data:`TITLE_BOOST_REPEATS` times followed by
            the article text.
        """
        title_block = " ".join([article.title] * TITLE_BOOST_REPEATS)
        return f"{title_block} {article.text}".strip()

    def get_enriched_corpus(self) -> list[str]:
        """Enriched text of every article, in storage order (for dense encoding).

        Returns:
            One enriched string per article.
        """
        return [self.get_enriched_text(article) for article in self.articles]

    def get_tokenized_corpus(self) -> list[list[str]]:
        """Normalized token lists of the enriched corpus (BM25 input).

        Uses :func:`src.utils.tokenize` (lowercasing + punctuation removal +
        whitespace splitting) on each enriched article text.

        Returns:
            One token list per article, in storage order.
        """
        return [tokenize(text) for text in self.get_enriched_corpus()]
