"""слой работы с данными: загрузка статей, валидация и обогащение текста.

пайплайн: статьи из feather -> конвертация HTML в Markdown -> валидация
pydantic -> обогащённый текст (усиление заголовка) -> нормализованный корпус
токенов для лексического (BM25) и семантического индексов. здесь же общий
загрузчик feather-таблиц для калибровочного и тестового наборов запросов.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.utils import chunk_text, html_to_markdown, tokenize

logger = logging.getLogger("rag.dataset")

# шаблон документа с усилением заголовка для BM25 и би-энкодера: заголовок
# повторяется дважды ("Title" и "Topic"), чтобы его слова весили больше в
# частотах термов и эмбеддингах, а маркеры полей задают явную структуру документа
ENRICHED_TEXT_TEMPLATE = "Title: {title} | Topic: {title} | Content: {body}"

# шаблон чанка для плотного индекса: каждый чанк несёт заголовок статьи,
# чтобы энкодер сохранял контекст документа после разбиения тела
CHUNK_TEXT_TEMPLATE = "Title: {title} | Content: {chunk}"


class Article(BaseModel):
    article_id: int
    title: str
    text: str


def load_feather_table(file_path: str | Path, required_columns: Sequence[str]) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Feather file not found: {path}")
    with warnings.catch_warnings():
        # pandas вызывает pyarrow.feather.read_feather, устаревший в pyarrow 24
        # в пользу IPC-ридера; глушим предупреждение, пока pandas не мигрирует
        warnings.simplefilter("ignore", FutureWarning)
        df = pd.read_feather(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}; found {list(df.columns)}")
    logger.info("загружено %d строк из %s (колонки: %s)", len(df), path, list(df.columns))
    return df


def sample_table(
    df: pd.DataFrame,
    sample_frac: float | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_frac is None and sample_size is None:
        return df
    if sample_size is not None:
        sampled = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    else:
        sampled = df.sample(frac=sample_frac, random_state=random_state)
    sampled = sampled.sort_index().reset_index(drop=True)
    logger.info(
        "отобрано %d из %d строк (frac=%s, size=%s, random_state=%d)",
        len(sampled),
        len(df),
        sample_frac,
        sample_size,
        random_state,
    )
    return sampled


class ArticleDataset:
    def __init__(self) -> None:
        self.articles: list[Article] = []
        # мемоизированный обогащённый корпус (вход токенизации BM25);
        # рендерится не более одного раза на загруженный корпус
        self._enriched_corpus: list[str] | None = None
        # мемоизированный чанкованный корпус с ключом по паре
        # (chunk_size, chunk_overlap), с которой он был построен
        self._chunked_corpus: tuple[list[str], list[int]] | None = None
        self._chunk_params: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.articles)

    def load_from_feather(self, file_path: str | Path) -> None:
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
        self._chunked_corpus = None
        self._chunk_params = None
        logger.info("распарсено %d статей (HTML -> Markdown) из %s", len(self.articles), file_path)

    def get_enriched_text(self, article: Article) -> str:
        return ENRICHED_TEXT_TEMPLATE.format(title=article.title, body=article.text)

    def get_enriched_corpus(self) -> list[str]:
        if self._enriched_corpus is None:
            self._enriched_corpus = [self.get_enriched_text(article) for article in self.articles]
        return self._enriched_corpus

    def get_chunked_corpus(self, chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[int]]:
        if self._chunked_corpus is None or self._chunk_params != (chunk_size, chunk_overlap):
            chunks: list[str] = []
            parents: list[int] = []
            for article in self.articles:
                for body_chunk in chunk_text(article.text, chunk_size, chunk_overlap) or [""]:
                    chunks.append(CHUNK_TEXT_TEMPLATE.format(title=article.title, chunk=body_chunk))
                    parents.append(article.article_id)
            self._chunked_corpus = (chunks, parents)
            self._chunk_params = (chunk_size, chunk_overlap)
            logger.info(
                "нарезано %d статей на %d чанков (size=%d, overlap=%d)",
                len(self.articles),
                len(chunks),
                chunk_size,
                chunk_overlap,
            )
        return self._chunked_corpus

    def get_tokenized_corpus(self) -> list[list[str]]:
        return [tokenize(text) for text in self.get_enriched_corpus()]
