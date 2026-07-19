"""точка входа пакетного пайплайна поиска.

автоматизирует полный цикл: загрузка статей из feather (очистка HTML) ->
построение или загрузка гибридного индекса -> валидация MAP@10 на
калибровочном наборе -> ранжированные топ-10 предсказания для тестового
набора -> экспорт ``answer.csv``.

все параметры берутся из дерева конфигов hydra (``configs/``), валидируемого
в :class:`src.config.AppConfig`; любое значение можно переопределить в
командной строке.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import AppConfig
from src.dataset import ArticleDataset, load_feather_table, sample_table
from src.indexer import HybridIndexer
from src.predict import predict
from src.searcher import HybridSearcher
from src.utils import calculate_map_at_10, set_seed, setup_logger

logger = logging.getLogger("rag.main")


def build_encoder(config: AppConfig) -> HybridIndexer:
    return HybridIndexer(
        model_name=config.model.bi_encoder,
        batch_size=config.model.batch_size,
        device=config.model.device,
        max_seq_length=config.model.max_seq_length,
        normalize_embeddings=config.model.normalize_embeddings,
        chunk_size=config.model.chunk_size,
        chunk_overlap=config.model.chunk_overlap,
    )


def ensure_index(dataset: ArticleDataset, encoder: HybridIndexer, config: AppConfig) -> None:
    if config.path.bm25_index.exists() and config.path.faiss_index.exists():
        logger.info(
            "найдены существующие артефакты индексов (%s, %s): построение пропущено",
            config.path.bm25_index,
            config.path.faiss_index,
        )
        return
    logger.info("артефакты индексов отсутствуют: строим по %d статьям", len(dataset))
    encoder.build_index(dataset)
    encoder.save(config.path.bm25_index, config.path.faiss_index)


def build_searcher(dataset: ArticleDataset, config: AppConfig) -> HybridSearcher:
    encoder = build_encoder(config)
    searcher_kwargs = {
        "dataset": dataset,
        "encoder": encoder,
        "bm25_path": config.path.bm25_index,
        "faiss_path": config.path.faiss_index,
        "rrf_k": config.hybrid.rrf_k,
        "bm25_weight": config.hybrid.bm25_weight,
        "dense_weight": config.hybrid.dense_weight,
        "aggregation_strategy": config.aggregation.strategy,
        "reranker_enabled": config.reranker.enabled,
        "reranker_name": config.reranker.model_name,
        "rerank_depth": config.reranker.rerank_depth,
        "device": config.model.device,
    }
    ensure_index(dataset, encoder, config)
    try:
        return HybridSearcher(**searcher_kwargs)
    except ValueError:
        logger.warning("артефакты индексов не соответствуют текущему корпусу: перестраиваем")
        config.path.bm25_index.unlink(missing_ok=True)
        config.path.faiss_index.unlink(missing_ok=True)
        ensure_index(dataset, encoder, config)
        return HybridSearcher(**searcher_kwargs)


def run_validation(searcher: HybridSearcher, config: AppConfig) -> None:
    if not config.path.calibration.exists():
        logger.warning("файл калибровки %s не найден: валидация пропущена", config.path.calibration)
        return
    calibration = load_feather_table(
        config.path.calibration, required_columns=("query_id", "query_text", "ground_truth")
    )
    calibration = sample_table(
        calibration,
        sample_frac=config.sampling.sample_frac,
        sample_size=config.sampling.sample_size,
        random_state=config.sampling.random_state,
    )
    predictions_table = predict(
        calibration,
        searcher,
        top_k=config.top_k_final,
        top_k_candidates=config.top_k_candidates,
        desc="calibration",
    )
    # сохраняем предсказания калибровки в формате сабмита для анализа ошибок
    # по каждому запросу относительно разметки
    config.path.calibration_answer.parent.mkdir(parents=True, exist_ok=True)
    predictions_table.to_csv(config.path.calibration_answer, index=False)
    logger.info("предсказания калибровки записаны в %s", config.path.calibration_answer)
    # строки ответов формата сабмита парсятся обратно в списки id — так же,
    # как официальный скорер читает answer.csv (тот же цикл сериализация -> парсинг)
    predictions = [[int(token) for token in answer.split()] for answer in predictions_table["answer"]]
    ground_truths = [[int(token) for token in str(truth).split()] for truth in calibration["ground_truth"]]
    score = calculate_map_at_10(predictions, ground_truths)
    logger.info("метрика MAP@10 на калибровке по %d запросам: %.4f", len(calibration), score)
    print(f"метрика MAP@10 на калибровке ({len(calibration)} запросов): {score:.4f}")


def run_test(searcher: HybridSearcher, config: AppConfig) -> None:
    # сэмплирование здесь намеренно НЕ применяется: сабмит обязан покрывать
    # каждый тестовый запрос, иначе выброшенные получат ноль
    test = load_feather_table(config.path.test, required_columns=("query_id", "query_text"))
    submission = predict(
        test,
        searcher,
        top_k=config.top_k_final,
        top_k_candidates=config.top_k_candidates,
        desc="test",
    )
    if len(submission) != len(test):
        raise RuntimeError(f"Submission has {len(submission)} rows for {len(test)} test queries")
    config.path.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.path.submission, index=False)
    logger.info("сабмит из %d строк записан в %s", len(submission), config.path.submission)
    print(f"сабмит записан в {config.path.submission} ({len(submission)} строк)")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    set_seed(config.seed)
    setup_logger(log_file=config.path.data_dir / "app.log")
    logger.info("итоговый конфиг:\n%s", OmegaConf.to_yaml(cfg))

    dataset = ArticleDataset()
    dataset.load_from_feather(config.path.articles)

    searcher = build_searcher(dataset, config)
    run_validation(searcher, config)
    run_test(searcher, config)


if __name__ == "__main__":
    main()
