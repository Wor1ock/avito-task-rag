"""CLI to inspect the exact preprocessed text a given article is indexed with.

Loads the corpus through the same pipeline as ``main.py`` (Feather ->
HTML-to-Markdown -> title-boosted enrichment) and prints, for each requested
``article_id``, the exact string fed to the BM25 tokenizer and the bi-encoder.
Intended for qualitative debugging of the preprocessing stage.

Run from the project root:

    PYTHONUTF8=1 uv run python -m src.inspect_article 1870
    PYTHONUTF8=1 uv run python -m src.inspect_article 1870 1951 --tokens
"""

from __future__ import annotations

import argparse
import logging

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.config import AppConfig
from src.dataset import ArticleDataset
from src.utils import tokenize

logger = logging.getLogger("rag.inspect_article")

SEPARATOR_WIDTH = 100


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Compose the Hydra config tree and validate it into :class:`AppConfig`.

    Args:
        overrides: Optional Hydra override strings (e.g. ``path.data_dir=mock_data``).

    Returns:
        Validated application config.
    """
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))


def print_article(dataset: ArticleDataset, article_id: int, show_tokens: bool = False) -> bool:
    """Print the enriched (indexed) representation of one article.

    Args:
        dataset: Loaded article dataset.
        article_id: Article to inspect.
        show_tokens: Also print the normalized BM25 token list.

    Returns:
        True if the article was found, False otherwise.
    """
    article = next((a for a in dataset.articles if a.article_id == article_id), None)
    if article is None:
        print(f"article_id={article_id}: not found in the corpus ({len(dataset)} articles)")
        return False
    enriched = dataset.get_enriched_text(article)
    print("=" * SEPARATOR_WIDTH)
    print(f"article_id={article.article_id} | title={article.title!r} | enriched length={len(enriched)} chars")
    print("=" * SEPARATOR_WIDTH)
    print(enriched)
    if show_tokens:
        tokens = tokenize(enriched)
        print("-" * SEPARATOR_WIDTH)
        print(f"BM25 tokens ({len(tokens)}):")
        print(" ".join(tokens))
    print()
    return True


def main() -> None:
    """Parse CLI arguments, load the corpus, and print the requested articles."""
    parser = argparse.ArgumentParser(
        description="Print the exact preprocessed (Markdown + title-boosted) text an article is indexed with."
    )
    parser.add_argument("article_ids", type=int, nargs="+", help="article_id values to inspect")
    parser.add_argument("--tokens", action="store_true", help="also print the normalized BM25 token list")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra config override (repeatable), e.g. --override path.data_dir=mock_data",
    )
    args = parser.parse_args()

    config = load_config(args.override)
    dataset = ArticleDataset()
    dataset.load_from_feather(config.path.articles)

    missing = [article_id for article_id in args.article_ids if not print_article(dataset, article_id, args.tokens)]
    if missing:
        raise SystemExit(f"Missing article ids: {missing}")


if __name__ == "__main__":
    main()
