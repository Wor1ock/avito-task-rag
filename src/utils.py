"""Evaluation metrics, text preprocessing, logging, and small shared helpers."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# Matches any character that is not a word character (letters/digits/underscore,
# Unicode-aware, so Cyrillic is preserved) and not whitespace — i.e. punctuation.
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_LOG_FILE = Path("data/app.log")


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


def average_precision_at_k(relevant: set[int], predicted: list[int], k: int = 10) -> float:
    """Average Precision at ``k`` for a single query.

    AP@k = (1 / min(|relevant|, k)) * sum_{i=1..k} P(i) * rel(i), where
    ``P(i)`` is precision at cutoff ``i`` and ``rel(i)`` indicates whether
    the item at rank ``i`` is relevant.

    Args:
        relevant: Ground-truth relevant document ids.
        predicted: Ranked predicted document ids (best first).
        k: Rank cutoff.

    Returns:
        AP@k in [0, 1]; 0.0 when ``relevant`` is empty.
    """
    raise NotImplementedError


def map_at_k(
    ground_truth: dict[int, set[int]],
    predictions: dict[int, list[int]],
    k: int = 10,
) -> float:
    """Mean Average Precision at ``k`` over a query set (MAP@10 by default).

    Args:
        ground_truth: Query id -> set of relevant document ids.
        predictions: Query id -> ranked predicted document ids.
        k: Rank cutoff.

    Returns:
        Mean of AP@k over all queries present in ``ground_truth``; queries
        missing from ``predictions`` contribute 0.
    """
    raise NotImplementedError


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Seed value.
    """
    raise NotImplementedError
