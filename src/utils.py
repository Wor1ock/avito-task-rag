"""метрики качества, предобработка текста, логирование и общие вспомогательные функции."""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

import html2text
import nltk
import numpy as np
import pymorphy3
import torch
from nltk.corpus import stopwords

# матчит любой символ, который не буква/цифра/подчёркивание (с поддержкой
# Unicode, поэтому кириллица сохраняется) и не пробел — то есть пунктуацию
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
# элементы script/style удаляются вместе с содержимым до конвертации в
# Markdown, иначе их текст просочился бы в результат
_HTML_INVISIBLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# три и более подряд идущих переноса строки схлопываются в одну пустую строку
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

logger = logging.getLogger("rag.utils")

DEFAULT_LOG_FILE = Path("data/app.log")
# кастомное отображение лемма -> кластерный токен, применяется только в
# токенизации BM25 (плотная ветка и кросс-энкодер видят сырой текст)
DEFAULT_SYNONYMS_FILE = Path("data/synonyms.json")
# проектные стоп-слова, объединяемые с русским списком NLTK для токенизации
# BM25 (одно слово на строку)
DEFAULT_STOPWORDS_FILE = Path("data/stopwords.txt")

# морфологический анализатор тяжёлый (грузит русские словари), поэтому
# создаётся лениво при первой токенизации, а не на импорте
_morph_analyzer: pymorphy3.MorphAnalyzer | None = None


@lru_cache(maxsize=1)
def russian_stop_words() -> frozenset[str]:
    try:
        words = stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("russian")
    return frozenset(words)


@lru_cache(maxsize=1)
def custom_stop_words() -> frozenset[str]:
    if not DEFAULT_STOPWORDS_FILE.exists():
        logger.warning("файл стоп-слов %s не найден: кастомные стоп-слова BM25 отключены", DEFAULT_STOPWORDS_FILE)
        return frozenset()
    words = DEFAULT_STOPWORDS_FILE.read_text(encoding="utf-8").split()
    logger.info("загружено %d кастомных стоп-слов из %s", len(words), DEFAULT_STOPWORDS_FILE)
    return frozenset(word.lower() for word in words)


@lru_cache(maxsize=1)
def bm25_stop_words() -> frozenset[str]:
    return russian_stop_words() | custom_stop_words()


@lru_cache(maxsize=1)
def synonym_map() -> dict[str, str]:
    if not DEFAULT_SYNONYMS_FILE.exists():
        logger.warning("файл синонимов %s не найден: отображение синонимов BM25 отключено", DEFAULT_SYNONYMS_FILE)
        return {}
    with DEFAULT_SYNONYMS_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info("загружено %d синонимических отображений из %s", len(mapping), DEFAULT_SYNONYMS_FILE)
    return {str(token).lower(): str(cluster).lower() for token, cluster in mapping.items()}


def _build_markdown_converter() -> html2text.HTML2Text:
    converter = html2text.HTML2Text()
    converter.body_width = 0  # без жёсткого переноса строк посреди предложения
    converter.ignore_links = True  # текст ссылки остаётся, URL выкидывается (шум для индекса)
    converter.ignore_images = True
    converter.ignore_emphasis = True  # убираем маркеры */_; заголовки/списки/таблицы остаются
    converter.ul_item_mark = "-"
    return converter


def html_to_markdown(html_text: str) -> str:
    text = _HTML_INVISIBLE_RE.sub(" ", html_text)
    markdown = _build_markdown_converter().handle(text)
    return _EXCESS_NEWLINES_RE.sub("\n\n", markdown).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        # окно достигло конца текста: следующие старты дали бы лишь суффиксы
        # этого чанка
        if start + chunk_size >= len(text):
            break
    return chunks


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=262_144)
def lemmatize(token: str) -> str:
    global _morph_analyzer
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy3.MorphAnalyzer()
    return _morph_analyzer.parse(token)[0].normal_form


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    stop_words = bm25_stop_words()
    synonyms = synonym_map()
    tokens: list[str] = []
    for token in normalized.split():
        if token in stop_words:
            continue
        lemma = lemmatize(token)
        if lemma not in stop_words:
            tokens.append(synonyms.get(lemma, lemma))
    return tokens


def set_seed(seed: int) -> None:
    random.seed(seed)
    # сторонние библиотеки читают легаси-глобальное состояние ГСЧ NumPy,
    # поэтому здесь нужен легаси-сидер, а не локальный Generator
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def setup_logger(
    name: str = "rag",
    log_file: str | Path = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def average_precision_at_10(predicted: list[int], relevant: list[int], k: int = 10) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in relevant_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_set), k)


def calculate_map_at_10(predictions: list[list[int]], ground_truths: list[list[int]]) -> float:
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Got {len(predictions)} predictions for {len(ground_truths)} ground truths")
    if not predictions:
        return 0.0
    ap_sum = sum(
        average_precision_at_10(predicted, relevant)
        for predicted, relevant in zip(predictions, ground_truths, strict=True)
    )
    return ap_sum / len(predictions)
