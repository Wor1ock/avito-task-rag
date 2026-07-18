"""Data management layer: article loading, validation, and text enrichment.

Pipeline: Feather articles -> HTML-to-Markdown conversion -> pydantic
validation -> enriched text (title boosting) -> normalized token corpus for
lexical (BM25) and semantic indexing. Also hosts the shared Feather table
loader used for the calibration and test query sets.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.utils import html_to_markdown, tokenize

logger = logging.getLogger("rag.dataset")

# Title-boosted document template fed to both BM25 tokenization and the
# bi-encoder: the title appears twice (as "Title" and "Topic") so its words
# carry more weight in term frequencies and embeddings, and the field markers
# give the encoder an explicit document structure.
ENRICHED_TEXT_TEMPLATE = "Title: {title} | Topic: {title} | Content: {body}"


class Article(BaseModel):
    """A single help-center article."""

    article_id: int
    title: str
    text: str


def load_feather_table(file_path: str | Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Load a Feather table and validate its schema.

    Args:
        file_path: Path to the ``.f`` / ``.feather`` file.
        required_columns: Columns that must be present.

    Returns:
        The loaded dataframe.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If any required column is missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Feather file not found: {path}")
    with warnings.catch_warnings():
        # pandas delegates to pyarrow.feather.read_feather, deprecated in pyarrow 24
        # in favor of the IPC reader; silence the noise until pandas migrates.
        warnings.simplefilter("ignore", FutureWarning)
        df = pd.read_feather(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}; found {list(df.columns)}")
    logger.info("Loaded %d rows from %s (columns: %s)", len(df), path, list(df.columns))
    return df


def sample_table(
    df: pd.DataFrame,
    sample_frac: float | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Reproducibly subsample a query table (calibration/test) for faster runs.

    A fixed ``random_state`` guarantees the same rows are selected on every
    run, so validation numbers stay comparable across iterations. Row order
    of the original table is preserved. Never apply this to the articles
    corpus — the persisted indexes cover the full corpus.

    Args:
        df: Table to subsample.
        sample_frac: Fraction of rows to keep, in (0, 1]. Ignored when
            ``sample_size`` is given.
        sample_size: Absolute number of rows to keep (capped at ``len(df)``);
            takes precedence over ``sample_frac``.
        random_state: Seed for pandas' sampler.

    Returns:
        The sampled table with a reset index, or ``df`` unchanged when both
        ``sample_frac`` and ``sample_size`` are None.
    """
    if sample_frac is None and sample_size is None:
        return df
    if sample_size is not None:
        sampled = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    else:
        sampled = df.sample(frac=sample_frac, random_state=random_state)
    sampled = sampled.sort_index().reset_index(drop=True)
    logger.info(
        "Sampled %d of %d rows (frac=%s, size=%s, random_state=%d)",
        len(sampled),
        len(df),
        sample_frac,
        sample_size,
        random_state,
    )
    return sampled


class ArticleDataset:
    """In-memory article store with validation and enrichment helpers."""

    def __init__(self) -> None:
        self.articles: list[Article] = []
        # Memoized enriched corpus; rendered at most once per loaded corpus
        # (shared by BM25 tokenization, dense encoding, and the reranker).
        self._enriched_corpus: list[str] | None = None

    def __len__(self) -> int:
        return len(self.articles)

    def load_from_feather(self, file_path: str | Path) -> None:
        """Load articles from a Feather file, converting the HTML ``body`` to Markdown.

        Expects ``article_id``, ``title``, and ``body`` columns; ``body`` is
        passed through :func:`src.utils.html_to_markdown` before validation,
        so the indexing corpuses are built over structured Markdown text
        (lists and tables keep their layout).

        Args:
            file_path: Path to the articles ``.f`` file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If required columns are missing.
            pydantic.ValidationError: If any row fails schema validation.
        """
        df = load_feather_table(file_path, required_columns=("article_id", "title", "body"))
        self.articles = [
            Article(
                article_id=int(row.article_id),
                title=str(row.title),
                text=html_to_markdown(str(row.body)),
            )
            for row in df.itertuples(index=False)
        ]
        self._enriched_corpus = None
        logger.info("Parsed %d articles (HTML -> Markdown) from %s", len(self.articles), file_path)

    def get_enriched_text(self, article: Article) -> str:
        """Render the title-boosted document string fed to both indexes.

        Args:
            article: Validated article.

        Returns:
            :data:`ENRICHED_TEXT_TEMPLATE` filled with the article's title
            (twice) and its Markdown body.
        """
        return ENRICHED_TEXT_TEMPLATE.format(title=article.title, body=article.text)

    def get_enriched_corpus(self) -> list[str]:
        """Enriched text of every article, in storage order (for dense encoding).

        The result is memoized: repeated calls (BM25 tokenization, dense
        encoding, reranker candidate texts) reuse the same rendered list until
        a new corpus is loaded.

        Returns:
            One enriched string per article.
        """
        if self._enriched_corpus is None:
            self._enriched_corpus = [self.get_enriched_text(article) for article in self.articles]
        return self._enriched_corpus

    def get_tokenized_corpus(self) -> list[list[str]]:
        """Normalized token lists of the enriched corpus (BM25 input).

        Uses :func:`src.utils.tokenize` (lowercasing + punctuation removal +
        Russian stop-word filtering + lemmatization) on each enriched article
        text.

        Returns:
            One token list per article, in storage order.
        """
        return [tokenize(text) for text in self.get_enriched_corpus()]
