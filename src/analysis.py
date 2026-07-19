"""анализ ошибок качества поиска на калибровочном наборе.

объединяет сохранённые предсказания калибровки (формат сабмита, их пишет
``main.py``) с разметкой, пересчитывает AP@10 по каждому запросу той же
реализацией метрики и печатает агрегированную статистику плюс худшие запросы
для ручного качественного разбора.
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
    return [int(token) for token in str(value).split()]


def build_analysis_table(calibration: pd.DataFrame, predictions: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    calibration = calibration.assign(query_id=calibration["query_id"].astype(int))
    predictions = predictions.assign(
        query_id=predictions["query_id"].astype(int),
        answer=predictions["answer"].fillna("").astype(str),
    )
    merged = calibration.merge(predictions, on="query_id", how="inner", validate="one_to_one")
    if len(merged) < len(calibration):
        logger.warning(
            "предсказания покрывают %d из %d калибровочных запросов: анализируется только пересечение",
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
                # ошибочные id в порядке ранжирования, то есть шум, вытолкнутый наверх
                "top_false_positives": [article_id for article_id in predicted if article_id not in truth_set],
            }
        )
    return pd.DataFrame(rows).sort_values(["ap_10", "query_id"]).reset_index(drop=True)


def print_report(analysis: pd.DataFrame, worst_n: int = WORST_QUERIES_TO_SHOW) -> None:
    total = len(analysis)
    if total == 0:
        print("нет пересечения запросов между предсказаниями и калибровкой: анализировать нечего")
        return
    map_10 = float(analysis["ap_10"].mean())
    failures = int((analysis["ap_10"] == 0.0).sum())
    perfect = int((analysis["ap_10"] == 1.0).sum())

    print("=" * 100)
    print("анализ ошибок на калибровке")
    print("=" * 100)
    print(f"всего проанализировано   : {total}")
    print(f"общая метрика MAP@10     : {map_10:.4f}")
    print(f"полные провалы (AP=0)    : {failures} ({failures / total:.1%})")
    print(f"идеальные ответы (AP=1)  : {perfect} ({perfect / total:.1%})")
    print()
    print(f"топ {min(worst_n, total)} худших запросов (наименьший AP@10)")
    print("-" * 100)
    for position, row in enumerate(analysis.head(worst_n).itertuples(index=False), start=1):
        text = row.query_text.replace("\n", " ")
        if len(text) > QUERY_TEXT_PREVIEW_CHARS:
            text = text[: QUERY_TEXT_PREVIEW_CHARS - 3] + "..."
        header = (
            f"#{position:>2} | query_id={row.query_id} | AP@10={row.ap_10:.3f}"
            f" | найдено {row.num_found}/{len(row.ground_truth_ids)}"
        )
        print(header)
        print(f"     запрос               : {text}")
        print(f"     релевантные id       : {row.ground_truth_ids}")
        print(f"     пропущенные id       : {row.missed_ids}")
        print(f"     ложные срабатывания  : {row.top_false_positives[:FALSE_POSITIVES_TO_SHOW]}")
        print("-" * 100)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
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
    logger.info("загружено %d предсказаний из %s", len(predictions), config.path.calibration_answer)

    analysis = build_analysis_table(calibration, predictions, top_k=config.top_k_final)
    print_report(analysis)


if __name__ == "__main__":
    main()
