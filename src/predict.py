"""пакетный пайплайн предсказаний: таблица запросов -> таблица предсказаний в формате сабмита.

общий шаг для валидации на калибровке и инференса на тесте, поэтому локальный
MAP@10 считается на той же структуре таблицы, что уходит в сабмит.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src.searcher import HybridSearcher

logger = logging.getLogger("rag.predict")


def predict(
    df: pd.DataFrame,
    searcher: HybridSearcher,
    top_k: int = 10,
    top_k_candidates: int = 100,
    desc: str = "predict",
) -> pd.DataFrame:
    missing = [column for column in ("query_id", "query_text") if column not in df.columns]
    if missing:
        raise ValueError(f"Query dataframe is missing required columns {missing}; found {list(df.columns)}")

    answers: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc):
        ranked = searcher.search(str(row.query_text), top_k_candidates=top_k_candidates, top_k_final=top_k)
        # защитная дедупликация с сохранением порядка; searcher уже возвращает уникальные id
        unique_ranked = list(dict.fromkeys(ranked))[:top_k]
        answers.append(" ".join(str(article_id) for article_id in unique_ranked))

    predictions = pd.DataFrame({"query_id": df["query_id"].astype(int).to_numpy(), "answer": answers})
    logger.info("построены топ-%d ранжирования для %d запросов", top_k, len(predictions))
    return predictions
