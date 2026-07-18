"""Evaluation metrics, text preprocessing, logging, and small shared helpers."""

from __future__ import annotations

import logging
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

import html2text
import numpy as np
import pymorphy3
import torch

# Matches any character that is not a word character (letters/digits/underscore,
# Unicode-aware, so Cyrillic is preserved) and not whitespace — i.e. punctuation.
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Script/style elements are dropped with their contents before the Markdown
# conversion, which would otherwise leak their text into the output.
_HTML_INVISIBLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# Three or more consecutive newlines collapse to one blank line.
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

DEFAULT_LOG_FILE = Path("data/app.log")

# Common Russian stop-words (function words carrying no retrieval signal).
# Tokens are checked against this set both as-is and after lemmatization, so
# base forms here (я, весь, этот, ...) also filter their inflected variants.
RUSSIAN_STOP_WORDS = frozenset(
    # A whitespace-delimited block stays readable and diff-friendly for a
    # 150-word list, unlike the one-quoted-string-per-word literal SIM905 wants.
    """
    а бы был была были было быть в вам вас вдруг ведь во вот впрочем все всегда
    всего всех всю вы г где да даже два для до другой его ее ей ему если есть
    еще ж же за зачем здесь и из или им иногда их к как какая какой когда
    конечно который кто куда ли лучше между меня мне много может можно мой моя
    мы на над надо наконец нас не него нее ней нельзя нет ни нибудь никогда ним
    них ничего но ну о об один он она они оно опять от перед по под после потом
    потому почти при про раз разве с сам свой себе себя сейчас со совсем так
    такой там тебя тем теперь то тогда того тоже только том тот три тут ты у
    уж уже хоть хорошо чего чем через что чтоб чтобы чуть эти этого этой этом
    этот эту я
    """.split()  # noqa: SIM905
)

# The morphological analyzer is heavyweight (loads the Russian dictionaries),
# so it is created lazily on first tokenization, not at import time.
_morph_analyzer: pymorphy3.MorphAnalyzer | None = None


def _build_markdown_converter() -> html2text.HTML2Text:
    """Fresh html2text converter (the handler is stateful, so one per call)."""
    converter = html2text.HTML2Text()
    converter.body_width = 0  # no hard line wrapping mid-sentence
    converter.ignore_links = True  # keep anchor text, drop URLs (index noise)
    converter.ignore_images = True
    converter.ignore_emphasis = True  # drop */_ markers; keep headers/lists/tables
    converter.ul_item_mark = "-"
    return converter


def html_to_markdown(html_text: str) -> str:
    """Convert article HTML to Markdown, preserving list and table layout.

    Script/style blocks are removed with their contents, then html2text
    renders the remaining markup as Markdown: headers, bullet/numbered lists,
    and tables keep their structure (one item/row per line) instead of being
    flattened into a single undifferentiated line. HTML entities are unescaped
    by the converter.

    Args:
        html_text: Raw HTML string (plain text passes through unchanged).

    Returns:
        Markdown-formatted plain-text string.
    """
    text = _HTML_INVISIBLE_RE.sub(" ", html_text)
    markdown = _build_markdown_converter().handle(text)
    return _EXCESS_NEWLINES_RE.sub("\n\n", markdown).strip()


def normalize_text(text: str) -> str:
    """Lowercase text and strip punctuation, collapsing repeated whitespace.

    Args:
        text: Raw input string.

    Returns:
        Normalized string: lowercase, punctuation replaced by spaces,
        consecutive whitespace collapsed, leading/trailing whitespace removed.
    """
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=262_144)
def lemmatize(token: str) -> str:
    """Reduce a Russian token to its normal form (lemma) via pymorphy3.

    Results are memoized: corpus vocabulary repeats heavily, so most lookups
    hit the cache instead of the morphological analyzer. Non-Russian tokens
    (Latin words, digits) pass through effectively unchanged.

    Args:
        token: A single lowercase word token.

    Returns:
        The token's lemma.
    """
    global _morph_analyzer
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy3.MorphAnalyzer()
    return _morph_analyzer.parse(token)[0].normal_form


def tokenize(text: str) -> list[str]:
    """Split text into lemma tokens for BM25, dropping Russian stop-words.

    Pipeline: :func:`normalize_text` (lowercase, punctuation removal) ->
    whitespace split -> stop-word filter -> :func:`lemmatize` -> stop-word
    filter on the lemma (catches inflected forms of stop-words).

    Args:
        text: Raw input string.

    Returns:
        List of lowercase lemmas; empty list for empty/punctuation-only input.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    for token in normalized.split():
        if token in RUSSIAN_STOP_WORDS:
            continue
        lemma = lemmatize(token)
        if lemma not in RUSSIAN_STOP_WORDS:
            tokens.append(lemma)
    return tokens


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed: Seed value applied to all three generators.
    """
    random.seed(seed)
    # Third-party libraries read NumPy's legacy global RNG state, so the
    # legacy seeder (not a local Generator) is required here.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def setup_logger(
    name: str = "rag",
    log_file: str | Path = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes to both the console and a log file.

    Idempotent: repeated calls with the same ``name`` reuse the existing
    handlers instead of duplicating them.

    Args:
        name: Logger name (child loggers inherit its handlers).
        log_file: Destination log file; parent directories are created.
        level: Minimum log level.

    Returns:
        Configured logger instance.
    """
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
    """Average Precision at ``k`` for a single query.

    AP@k = (1 / min(|relevant|, k)) * sum over ranks i where the item is
    relevant of precision@i. Rewards placing all relevant articles as high
    as possible within the top ``k``.

    Args:
        predicted: Ranked predicted article ids (best first).
        relevant: Ground-truth relevant article ids.
        k: Rank cutoff.

    Returns:
        AP@k in [0, 1]; 0.0 when ``relevant`` is empty.
    """
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
    """Mean Average Precision at 10 over a query set.

    Args:
        predictions: Per-query ranked predicted article ids (best first).
        ground_truths: Per-query relevant article ids, aligned with ``predictions``.

    Returns:
        Mean of AP@10 over all queries; 0.0 for an empty input.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Got {len(predictions)} predictions for {len(ground_truths)} ground truths")
    if not predictions:
        return 0.0
    ap_sum = sum(
        average_precision_at_10(predicted, relevant)
        for predicted, relevant in zip(predictions, ground_truths, strict=True)
    )
    return ap_sum / len(predictions)
