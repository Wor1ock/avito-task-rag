"""утилита CLI: показывает точный предобработанный текст, под которым статья индексируется.

грузит корпус тем же пайплайном, что и ``main.py`` (feather -> HTML в Markdown
-> обогащение заголовком), и печатает для каждого запрошенного ``article_id``
ту строку, которая подаётся в токенизатор BM25 и би-энкодер. предназначена для
качественной отладки этапа предобработки.
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
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))


def print_article(dataset: ArticleDataset, article_id: int, show_tokens: bool = False) -> bool:
    article = next((a for a in dataset.articles if a.article_id == article_id), None)
    if article is None:
        print(f"article_id={article_id}: не найден в корпусе ({len(dataset)} статей)")
        return False
    enriched = dataset.get_enriched_text(article)
    print("=" * SEPARATOR_WIDTH)
    print(f"article_id={article.article_id} | title={article.title!r} | длина обогащённого текста={len(enriched)}")
    print("=" * SEPARATOR_WIDTH)
    print(enriched)
    if show_tokens:
        tokens = tokenize(enriched)
        print("-" * SEPARATOR_WIDTH)
        print(f"токены BM25 ({len(tokens)}):")
        print(" ".join(tokens))
    print()
    return True


def main() -> None:
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
