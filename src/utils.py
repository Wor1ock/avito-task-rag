"""Evaluation metrics, text preprocessing, logging, and small shared helpers."""

from __future__ import annotations

import html
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

# Matches any character that is not a word character (letters/digits/underscore,
# Unicode-aware, so Cyrillic is preserved) and not whitespace — i.e. punctuation.
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Script/style elements are dropped with their contents; every other tag
# (including tables, spoilers, and unclosed/broken tags) is replaced by a space.
_HTML_INVISIBLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

DEFAULT_LOG_FILE = Path("data/app.log")


def clean_html(html_text: str) -> str:
    """Strip HTML markup and return pure text (regex-based, zero-dependency).

    Removes script/style blocks entirely, replaces all remaining tags with
    spaces (so adjacent table cells do not merge into one token), unescapes
    HTML entities, and collapses redundant whitespace.

    Args:
        html_text: Raw HTML string (plain text passes through unchanged).

    Returns:
        Cleaned plain-text string.
    """
    text = _HTML_INVISIBLE_RE.sub(" ", html_text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


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


def tokenize(text: str) -> list[str]:
    """Split text into word tokens for BM25 (normalization + whitespace split).

    Args:
        text: Raw input string.

    Returns:
        List of lowercase tokens; empty list for empty/punctuation-only input.
    """
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


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
