# RAG Retrieval Baseline

Information Retrieval baseline for a RAG pipeline: hybrid search combining
**BM25** (sparse) and **FAISS** dense retrieval over a **LaBSE** bi-encoder,
fused with **Reciprocal Rank Fusion (RRF)** and evaluated with **MAP@10**.

## Pipeline

```
articles.f ──► clean HTML ──► chunk ──► BM25 index ─┐
                                   └──► FAISS index ─┤► RRF fusion ──► (re-ranker) ──► top-10 docs
queries ─────────────────────────────────────────────┘
```

## Project layout

```
configs/            Hydra configs (main + path/model groups)
src/dataset.py      Loading (Feather/Parquet), HTML cleaning, chunking
src/indexer.py      BM25Indexer and FaissIndexer (build/save/load)
src/searcher.py     HybridSearcher: BM25 + FAISS + RRF + re-ranker stub
src/utils.py        MAP@10 metric, seeding
main.py             Hydra entry point orchestrating the full baseline
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10.

```bash
uv sync                 # CPU FAISS (default)
uv sync --extra gpu     # + faiss-gpu-cu12 (Linux only)
```

PyTorch is resolved from the CUDA 12.4 wheel index (`download.pytorch.org/whl/cu124`)
on Linux/Windows automatically.

## Data

Place the datasets under `data/` (configurable in `configs/path/default.yaml`):

- `data/articles.f` — article corpus (HTML text)
- `data/calibration.f` — validation queries with relevance labels
- `data/test.f` — test queries for the submission

## Run

```bash
uv run python main.py                          # full baseline
uv run python main.py model.device=cpu         # override any config value
uv run python main.py top_k_candidates=200 hybrid.rrf_k=20
```

Hydra writes run logs to `outputs/` (git-ignored).

## Docker

```bash
docker build -t rag-retrieval-baseline .
docker run --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  rag-retrieval-baseline
```

## Status

This is a **skeleton**: all functions/classes carry full signatures and
docstrings but raise `NotImplementedError`. Implement in this order:

1. `src/dataset.py` — loading, cleaning, chunking
2. `src/indexer.py` — BM25 and FAISS build/save/load
3. `src/searcher.py` — retrieval, RRF fusion, doc aggregation
4. `src/utils.py` — AP@k / MAP@k
5. `main.py` — wire ground truth + submission writing to the real schema
