"""Evaluation metrics, text preprocessing, logging, and small shared helpers."""

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

# Matches any character that is not a word character (letters/digits/underscore,
# Unicode-aware, so Cyrillic is preserved) and not whitespace — i.e. punctuation.
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Script/style elements are dropped with their contents before the Markdown
# conversion, which would otherwise leak their text into the output.
_HTML_INVISIBLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# Three or more consecutive newlines collapse to one blank line.
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

logger = logging.getLogger("rag.utils")

DEFAULT_LOG_FILE = Path("data/app.log")
# Custom lemma -> canonical-cluster token mapping applied only in the BM25
# tokenization path (the dense and cross-encoder branches see raw text).
DEFAULT_SYNONYMS_FILE = Path("data/synonyms.json")

# The morphological analyzer is heavyweight (loads the Russian dictionaries),
# so it is created lazily on first tokenization, not at import time.
_morph_analyzer: pymorphy3.MorphAnalyzer | None = None


@lru_cache(maxsize=1)
def russian_stop_words() -> frozenset[str]:
    """NLTK's official Russian stop-word list, fetched lazily.

    The ``stopwords`` corpus is downloaded on first use when missing from the
    local NLTK data directory; afterwards the set is served from cache.
    Tokens are checked against this set both as-is and after lemmatization,
    so base forms in the list (я, весь, этот, ...) also filter their
    inflected variants.

    Returns:
        Frozen set of lowercase Russian stop-words.
    """
    try:
        words = stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("russian")
    return frozenset(words)


@lru_cache(maxsize=1)
def synonym_map() -> dict[str, str]:
    """Custom lemma -> canonical-cluster token mapping for BM25 tokenization.

    Loaded once from :data:`DEFAULT_SYNONYMS_FILE`. Keys must be lemmas: the
    mapping is applied after pymorphy3 lemmatization, collapsing domain
    synonyms (товар/заказ/посылка -> объект_сделки, ...) onto one shared
    token so BM25 matches across the whole cluster. Only the lexical branch
    uses it; dense and cross-encoder inputs stay untouched.

    Returns:
        Lowercase lemma -> cluster-token dict; empty when the file is missing.
    """
    if not DEFAULT_SYNONYMS_FILE.exists():
        logger.warning("Synonym file %s not found: BM25 synonym mapping disabled", DEFAULT_SYNONYMS_FILE)
        return {}
    with DEFAULT_SYNONYMS_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info("Loaded %d synonym mappings from %s", len(mapping), DEFAULT_SYNONYMS_FILE)
    return {str(token).lower(): str(cluster).lower() for token, cluster in mapping.items()}


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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into character windows of ``chunk_size`` sharing ``chunk_overlap``.

    Windows advance by ``chunk_size - chunk_overlap`` characters, so every
    chunk repeats the tail of its predecessor and sentences cut by a boundary
    stay intact in at least one chunk.

    Args:
        text: Input string.
        chunk_size: Window length in characters.
        chunk_overlap: Characters shared between consecutive windows.

    Returns:
        Stripped non-empty chunks in document order; empty list for blank input.

    Raises:
        ValueError: If ``chunk_overlap`` is not smaller than ``chunk_size``.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        # The window reached the end of the text: further starts would only
        # produce suffixes of this chunk.
        if start + chunk_size >= len(text):
            break
    return chunks


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
    whitespace split -> stop-word filter (:func:`russian_stop_words`) ->
    :func:`lemmatize` -> stop-word filter on the lemma (catches inflected
    forms of stop-words) -> synonym mapping (:func:`synonym_map`) onto the
    lemma's canonical cluster token.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase lemmas; empty list for empty/punctuation-only input.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    stop_words = russian_stop_words()
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
