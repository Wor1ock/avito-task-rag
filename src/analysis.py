"""Error analysis for calibration retrieval quality.

Joins the persisted calibration predictions (submission format, written by
``main.py``) with the calibration ground truth, recomputes per-query AP@10
with the exact metric implementation, and prints aggregate statistics plus
the worst-performing queries for manual qualitative inspection.

Run from the project root (after a ``main.py`` run has produced the
calibration predictions artifact):

    PYTHONUTF8=1 uv run python -m src.analysis
"""

from __future__ import annotations

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config import AppConfig
from src.dataset import load_feather_table
from src.utils import average_precision_at_10, setup_logger

logger = logging.getLogger("rag.analysis")

WORST_QUERIES_TO_SHOW = 15
FALSE_POSITIVES_TO_SHOW = 5
QUERY_TEXT_PREVIEW_CHARS = 120


def parse_id_string(value: str) -> list[int]:
    """Space-separated id string (submission ``answer`` format) -> int list."""
    return [int(token) for token in str(value).split()]


def build_analysis_table(calibration: pd.DataFrame, predictions: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Per-query error breakdown of predictions against the ground truth.

    Rows are joined on ``query_id`` (both sides cast to int), so a predictions
    file produced from a sampled calibration run is analyzed over exactly the
    queries it contains.

    Args:
        calibration: Ground-truth table with ``query_id``, ``query_text``,
            and ``ground_truth`` (space-separated relevant article ids).
        predictions: Submission-format table with ``query_id`` and ``answer``.
        top_k: Rank cutoff of the metric.

    Returns:
        One row per analyzed query, sorted by increasing ``ap_10`` (worst
        first, ties broken by ``query_id``), with columns: ``query_id``,
        ``query_text``, ``ap_10``, ``num_found``, ``ground_truth_ids``,
        ``predicted_ids``, ``missed_ids``, ``top_false_positives``.
    """
    calibration = calibration.assign(query_id=calibration["query_id"].astype(int))
    predictions = predictions.assign(
        query_id=predictions["query_id"].astype(int),
        answer=predictions["answer"].fillna("").astype(str),
    )
    merged = calibration.merge(predictions, on="query_id", how="inner", validate="one_to_one")
    if len(merged) < len(calibration):
        logger.warning(
            "Predictions cover %d of %d calibration queries: analyzing the intersection only",
            len(merged),
            len(calibration),
        )

    rows: list[dict] = []
    for row in merged.itertuples(index=False):
        truth = parse_id_string(row.ground_truth)
        predicted = parse_id_string(row.answer)[:top_k]
        truth_set = set(truth)
        predicted_set = set(predicted)
        rows.append(
            {
                "query_id": int(row.query_id),
                "query_text": str(row.query_text),
                "ap_10": average_precision_at_10(predicted, truth, k=top_k),
                "num_found": len(truth_set & predicted_set),
                "ground_truth_ids": truth,
                "predicted_ids": predicted,
                "missed_ids": [article_id for article_id in truth if article_id not in predicted_set],
                # Incorrect ids in ranking order, i.e. the noise pushed to the top.
                "top_false_positives": [article_id for article_id in predicted if article_id not in truth_set],
            }
        )
    return pd.DataFrame(rows).sort_values(["ap_10", "query_id"]).reset_index(drop=True)


def print_report(analysis: pd.DataFrame, worst_n: int = WORST_QUERIES_TO_SHOW) -> None:
    """Print aggregate statistics and the worst-performing queries.

    Args:
        analysis: Output of :func:`build_analysis_table` (sorted worst-first).
        worst_n: Number of lowest-AP@10 queries to detail.
    """
    total = len(analysis)
    if total == 0:
        print("No overlapping queries between predictions and calibration: nothing to analyze")
        return
    map_10 = float(analysis["ap_10"].mean())
    failures = int((analysis["ap_10"] == 0.0).sum())
    perfect = int((analysis["ap_10"] == 1.0).sum())

    print("=" * 100)
    print("CALIBRATION ERROR ANALYSIS")
    print("=" * 100)
    print(f"Total queries analyzed   : {total}")
    print(f"Overall MAP@10           : {map_10:.4f}")
    print(f"Complete failures (AP=0) : {failures} ({failures / total:.1%})")
    print(f"Perfect matches (AP=1)   : {perfect} ({perfect / total:.1%})")
    print()
    print(f"TOP {min(worst_n, total)} WORST QUERIES (lowest AP@10)")
    print("-" * 100)
    for position, row in enumerate(analysis.head(worst_n).itertuples(index=False), start=1):
        text = row.query_text.replace("\n", " ")
        if len(text) > QUERY_TEXT_PREVIEW_CHARS:
            text = text[: QUERY_TEXT_PREVIEW_CHARS - 3] + "..."
        header = (
            f"#{position:>2} | query_id={row.query_id} | AP@10={row.ap_10:.3f}"
            f" | found {row.num_found}/{len(row.ground_truth_ids)}"
        )
        print(header)
        print(f"     query               : {text}")
        print(f"     ground truth ids    : {row.ground_truth_ids}")
        print(f"     missed ids          : {row.missed_ids}")
        print(f"     top false positives : {row.top_false_positives[:FALSE_POSITIVES_TO_SHOW]}")
        print("-" * 100)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load predictions + ground truth, compute per-query AP@10, print the report.

    Args:
        cfg: Hydra-composed configuration (validated into :class:`AppConfig`).
    """
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    setup_logger(log_file=config.path.data_dir / "app.log")

    if not config.path.calibration_answer.exists():
        raise FileNotFoundError(
            f"Calibration predictions not found at {config.path.calibration_answer}; run main.py first"
        )
    calibration = load_feather_table(
        config.path.calibration, required_columns=("query_id", "query_text", "ground_truth")
    )
    predictions = pd.read_csv(config.path.calibration_answer)
    logger.info("Loaded %d predictions from %s", len(predictions), config.path.calibration_answer)

    analysis = build_analysis_table(calibration, predictions, top_k=config.top_k_final)
    print_report(analysis)


if __name__ == "__main__":
    main()
