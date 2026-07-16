"""Interactive hybrid search entry point.

Glues the full pipeline together: ensures a corpus exists (mock articles or a
generated dummy set), builds and saves the BM25 + FAISS indexes when the
artifacts are missing, then serves an interactive CLI loop that prints the
top-10 ranked ``article_id`` list with raw reranker scores.

Run from the project root:

    uv run python main.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.dataset import ArticleDataset
from src.indexer import BM25_FILENAME, FAISS_FILENAME, HybridIndexer
from src.searcher import HybridSearcher
from src.utils import setup_logger

logger = logging.getLogger("rag.main")

DATA_DIR = Path("data")
ARTICLES_PATH = DATA_DIR / "mock_articles.json"
EXIT_COMMAND = "exit"

# Fallback corpus written to ARTICLES_PATH when no articles file is present.
DUMMY_ARTICLES = [
    {
        "article_id": 1,
        "title": "Как восстановить доступ к аккаунту",
        "text": "Если вы забыли пароль, нажмите «Забыли пароль?» на странице входа и следуйте инструкциям.",
    },
    {
        "article_id": 2,
        "title": "Правила размещения объявлений",
        "text": "Объявление должно относиться к разрешённой категории. Дубликаты и недостоверные цены запрещены.",
    },
    {
        "article_id": 3,
        "title": "Как настроить доставку",
        "text": "Включите Авито Доставку в настройках объявления и выберите удобные пункты отправки товара.",
    },
    {
        "article_id": 4,
        "title": "Возврат товара и денег",
        "text": "Покупатель может открыть спор в течение периода защиты, если товар не соответствует описанию.",
    },
    {
        "article_id": 5,
        "title": "Как продвигать объявления",
        "text": "Платные услуги продвижения поднимают объявление в результатах поиска и увеличивают просмотры.",
    },
    {
        "article_id": 6,
        "title": "Безопасная сделка: как это работает",
        "text": "Деньги замораживаются на счёте до получения товара покупателем, после чего переводятся продавцу.",
    },
    {
        "article_id": 7,
        "title": "Что делать при встрече с мошенниками",
        "text": "Не переходите по внешним ссылкам и не сообщайте коды из СМС. Пожалуйтесь на подозрительный профиль.",
    },
    {
        "article_id": 8,
        "title": "Тарифы и оплата для бизнеса",
        "text": "Для профессиональных продавцов доступны пакеты размещений и кабинет Авито Про со статистикой.",
    },
]


def ensure_articles_file() -> None:
    """Create ``ARTICLES_PATH`` with the dummy corpus when it is missing."""
    if ARTICLES_PATH.exists():
        return
    ARTICLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTICLES_PATH.write_text(json.dumps(DUMMY_ARTICLES, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Articles file missing: generated dummy dataset at %s (%d articles)",
        ARTICLES_PATH,
        len(DUMMY_ARTICLES),
    )


def ensure_index(dataset: ArticleDataset) -> None:
    """Build and persist the hybrid index unless both artifacts already exist.

    Args:
        dataset: Corpus to index when the artifacts are missing.
    """
    bm25_path = DATA_DIR / BM25_FILENAME
    faiss_path = DATA_DIR / FAISS_FILENAME
    if bm25_path.exists() and faiss_path.exists():
        logger.info("Found existing index artifacts (%s, %s): skipping build", bm25_path, faiss_path)
        return
    logger.info("Index artifacts missing: building from %d articles", len(dataset))
    indexer = HybridIndexer()
    indexer.build_index(dataset)
    indexer.save(str(DATA_DIR))


def run_cli(searcher: HybridSearcher) -> None:
    """Serve queries interactively until the user types 'exit'.

    Args:
        searcher: Fully initialized hybrid searcher.
    """
    print("Hybrid help-center search. Type a question, or 'exit' to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() == EXIT_COMMAND:
            break
        results = searcher.search_with_scores(query)
        print(f"Top-{len(results)} articles:")
        for rank, (article_id, score) in enumerate(results, start=1):
            print(f"  {rank:2d}. article_id={article_id:<6d} reranker_score={score:+.4f}")
    print("Bye.")


def main() -> None:
    """Run the full pipeline: corpus -> (build or load) index -> CLI loop."""
    setup_logger()
    ensure_articles_file()

    dataset = ArticleDataset()
    dataset.load_from_json(str(ARTICLES_PATH))

    ensure_index(dataset)
    searcher = HybridSearcher(dataset, index_dir=str(DATA_DIR))
    run_cli(searcher)


if __name__ == "__main__":
    main()
