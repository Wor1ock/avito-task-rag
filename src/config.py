"""типизированная конфигурация приложения.

схемы pydantic v2 валидируют собранное hydra дерево конфигов (``configs/``)
до входа в пайплайн: каждый параметр проверяется один раз на старте и читается
через типизированные атрибуты вместо сырых ключей ``DictConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathConfig(BaseModel):
    data_dir: Path
    articles: Path
    calibration: Path
    test: Path
    index_dir: Path
    bm25_index: Path
    faiss_index: Path
    submission: Path
    calibration_answer: Path


class ModelConfig(BaseModel):
    bi_encoder: str
    device: str | None = None
    batch_size: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    normalize_embeddings: bool
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ModelConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class HybridConfig(BaseModel):
    rrf_k: float = Field(gt=0)
    bm25_weight: float = Field(ge=0)
    dense_weight: float = Field(ge=0)


class AggregationConfig(BaseModel):
    strategy: Literal["max_p", "avg_p", "sum_p"] = "max_p"


class RerankerConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool
    model_name: str | None = None
    rerank_depth: int = Field(default=15, gt=0)


class SamplingConfig(BaseModel):
    sample_frac: float | None = Field(default=None, gt=0, le=1)
    sample_size: int | None = Field(default=None, gt=0)
    random_state: int = 42

    @model_validator(mode="after")
    def _at_most_one_mode(self) -> SamplingConfig:
        if self.sample_frac is not None and self.sample_size is not None:
            raise ValueError("Set at most one of sample_frac or sample_size, not both")
        return self


class AppConfig(BaseModel):
    path: PathConfig
    model: ModelConfig
    top_k_candidates: int = Field(gt=0)
    top_k_final: int = Field(gt=0)
    hybrid: HybridConfig
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    reranker: RerankerConfig
    sampling: SamplingConfig
    seed: int
