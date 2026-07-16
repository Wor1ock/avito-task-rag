"""Typed application configuration.

Pydantic v2 schemas that validate the composed Hydra tree (``configs/``)
before it reaches the pipeline, so every parameter is checked once at startup
and accessed through typed attributes instead of raw ``DictConfig`` keys.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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


class AppConfig(BaseModel):
    """Root config composed by Hydra from ``configs/config.yaml``."""

    path: PathConfig
    model: ModelConfig
    top_k_candidates: int = Field(gt=0)
    top_k_final: int = Field(gt=0)
    hybrid: HybridConfig
    reranker: RerankerConfig
    seed: int
