"""Typed application configuration.

Pydantic v2 schemas that validate the composed Hydra tree (``configs/``)
before it reaches the pipeline, so every parameter is checked once at startup
and accessed through typed attributes instead of raw ``DictConfig`` keys.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathConfig(BaseModel):
    """Data and artifact locations (``configs/path``)."""

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
    """Bi-encoder settings (``configs/model``)."""

    bi_encoder: str
    device: str | None = None
    batch_size: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    normalize_embeddings: bool


class HybridConfig(BaseModel):
    """Weighted Reciprocal Rank Fusion parameters."""

    rrf_k: float = Field(gt=0)
    bm25_weight: float = Field(ge=0)
    dense_weight: float = Field(ge=0)


class RerankerConfig(BaseModel):
    """Optional cross-encoder re-ranking stage."""

    model_config = ConfigDict(protected_namespaces=())

    enabled: bool
    model_name: str | None = None


class SamplingConfig(BaseModel):
    """Reproducible subsampling of the calibration/test query sets."""

    sample_frac: float | None = Field(default=None, gt=0, le=1)
    sample_size: int | None = Field(default=None, gt=0)
    random_state: int = 42

    @model_validator(mode="after")
    def _at_most_one_mode(self) -> SamplingConfig:
        if self.sample_frac is not None and self.sample_size is not None:
            raise ValueError("Set at most one of sample_frac or sample_size, not both")
        return self


class AppConfig(BaseModel):
    """Root config composed by Hydra from ``configs/config.yaml``."""

    path: PathConfig
    model: ModelConfig
    top_k_candidates: int = Field(gt=0)
    top_k_final: int = Field(gt=0)
    hybrid: HybridConfig
    reranker: RerankerConfig
    sampling: SamplingConfig
    seed: int
