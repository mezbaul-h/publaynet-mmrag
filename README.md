# PubLayNet Multimodal RAG + Knowledge Graph

A retrieval-augmented question-answering system over the
[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) document-layout dataset. It
answers scientific questions by retrieving over both **text and figures/tables**,
augments retrieval with a small **knowledge graph**, generates **grounded, cited**
answers, and reports a quantitative comparison between a **text-only baseline** and
an **enhanced multimodal + graph** pipeline.

Everything runs locally through Python packages — no Docker, no external inference
server. Model weights download from the Hugging Face Hub on first use.

## Table of contents

- [What it does](#what-it-does)
- [System requirements](#system-requirements)
- [Setup](#setup)
- [Reproduce the results](#reproduce-the-results)
- [Results](#results)
- [Demo and API](#demo-and-api)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Advanced usage](#advanced-usage)
  - [Environment and PyTorch (CPU / GPU / CUDA)](#environment-and-pytorch-cpu--gpu--cuda)
  - [Optional extras](#optional-extras)
  - [Configuration and flags](#configuration-and-flags)
  - [GPU memory (VRAM) tuning](#gpu-memory-vram-tuning)
  - [Per-component ablations](#per-component-ablations)
  - [Larger or quantised generators](#larger-or-quantised-generators)
  - [Scaling to the full dataset](#scaling-to-the-full-dataset)
  - [Resumability and out-of-memory tuning](#resumability-and-out-of-memory-tuning)
  - [Development](#development)
  - [Version notes](#version-notes)

## What it does

You ask a natural-language question. The system retrieves the most relevant
evidence — text chunks, figure/table crops, and graph-connected chunks — fuses
them, and a language model writes a short answer that cites its sources. A
**baseline** arm uses text-only retrieval; an **enhanced** arm adds visual
retrieval, a knowledge graph, hybrid (dense + sparse) text search, and reranking.
A built-in evaluation scores both arms on three question types (text, visual,
multi-hop) so you can see exactly what each addition contributes.

## System requirements

A recent NVIDIA GPU is required, with **CUDA 13 or newer**. CUDA 12 builds do not
work here — current NVIDIA GPUs such as the RTX 50-series (Blackwell) need CUDA 13.
See [Environment and PyTorch](#environment-and-pytorch-cpu--gpu--cuda) for picking
the right PyTorch build.

| Component   | Recommended                         | Test system (this build)               |
|-------------|-------------------------------------|----------------------------------------|
| OS          | Linux (Ubuntu 22.04 / 24.04)        | Ubuntu 24.04.4 LTS (kernel 6.17)       |
| GPU         | NVIDIA, **≥ 12 GiB** VRAM           | GeForce RTX 5070, 12 GiB               |
| GPU driver  | **CUDA 13+** (CUDA 12 does not work)| driver 595.71.05, CUDA 13.2            |
| CPU         | 8 cores or more                     | AMD Ryzen 7 9700X (8c / 16t)           |
| RAM         | ≥ 16 GiB (32 GiB comfortable)       | 32 GiB                                 |
| Disk        | ~10 GiB free (models + index)       | —                                      |
| Python      | 3.11                                | 3.11.15                                |
| PyTorch     | build matching your CUDA            | 2.12.1+cu132                           |

The pipeline is split into stages so the models never all sit in VRAM at once; it
fits comfortably on a 12 GiB card.

## Setup

Pick **one** of the two options. Both install the project and all its
dependencies (including `torchvision`, used by the figure captioner). First
install a CUDA 13+ build of PyTorch (see
[Environment and PyTorch](#environment-and-pytorch-cpu--gpu--cuda) if unsure), then:

**Option A — pip + virtual environment**

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

**Option B — conda**

```bash
conda env create -f environment.yml
conda activate publaynet-mmrag
```

`conda env create` already installs the project and its pip dependencies. Model
weights are downloaded automatically the first time each stage runs.

## Reproduce the results

Activate your environment, then run these five commands in order from the project
root. They download the data subset and model weights on first use, and write the
final comparison to `data/results/comparison.json`.

```bash
# Stage 1 — download the dataset subset, crop regions, OCR the text.
python scripts/01_ingest_preprocess.py --mode enhanced

# Stage 1b — caption 500 figure/table crops (needed for the visual questions).
python scripts/01b_caption_regions.py

# Stage 2 — build the text + image vector index.
python scripts/02_build_index.py --mode enhanced

# Stage 3 — build the knowledge graph.
python scripts/03_build_kg.py --mode enhanced

# Stage 4 — evaluate every variant and write data/results/comparison.json.
python scripts/04_run_eval.py --variants baseline,abl_rerank,abl_hybrid,abl_image,abl_graph,enhanced
```

Each stage saves its output to disk and can be re-run safely — it skips work
already done. Stage 4 also builds the evaluation questions on first run (one text,
visual, and multi-hop set) and caches them in `data/qa.jsonl`.

## Results

From `data/results/comparison.json` (Recall@5; visual and multi-hop use exact
figure/chunk matching). The text-only baseline cannot retrieve a figure or the
multi-hop bridge chunks, so it scores 0 there; the enhanced arm recovers both.

| Question type  | Baseline | Enhanced |
|----------------|----------|----------|
| text           | 0.97     | 1.00     |
| visual         | 0.00     | 0.56     |
| multihop       | 0.00     | 0.73     |
| **overall**    | 0.66     | 0.87     |

The full analysis, per-component attribution, and generation-quality scores are in
[`report.md`](report.md).

## Demo and API

Both need an extra installed (see [Optional extras](#optional-extras)).

> **Important — build the data first.** The demo and the API query the prebuilt
> index and knowledge graph, so you must run the pipeline before starting them:
> **Stages 1, 1b, 2 and 3** (see [Reproduce the results](#reproduce-the-results)).
> Stage 1b is needed for visual answers and the figure gallery; the knowledge graph
> (Stage 3) is needed for the graph channel in `enhanced` mode. **Stage 4
> (evaluation) is not required.** Without the index/graph on disk, startup fails.

```bash
# Web UI (Gradio): question box, answer, evidence, and a figure gallery.
python app/demo.py

# HTTP API (FastAPI) on http://127.0.0.1:8000
python scripts/05_serve.py --mode enhanced
```

For visual questions, both answer with a vision-language model that reads the
retrieved figure/table crop directly, instead of paraphrasing a caption.

## How it works

The pipeline runs in stages; each loads only the models it needs, writes its output
to disk, and frees the GPU before the next.

```
Stage 1    ingest + crop + OCR                 -> page regions, text chunks, crops
Stage 1b   caption a sample of figure crops    -> captions (for the visual questions)
Stage 2    embed text (BGE-M3) + images (SigLIP2) -> vector index (Qdrant)
Stage 3    entities (GLiNER) + relations (LLM)  -> knowledge graph (NetworkX)
Stage 4    build questions, run both arms, score -> comparison.json
```

| Part            | Model / tool                              |
|-----------------|-------------------------------------------|
| OCR             | Surya                                     |
| Text embedding  | BAAI/bge-m3 (dense + sparse)              |
| Image embedding | google/siglip2-base-patch16-224           |
| Reranking       | BAAI/bge-reranker-v2-m3                    |
| Entities (NER)  | urchade/gliner_medium-v2.1                |
| Captioning / visual answers | Qwen/Qwen2.5-VL-3B-Instruct   |
| Text generation | Qwen/Qwen3-4B-Instruct-2507               |
| Vector store    | Qdrant (embedded, on-disk)                |
| Knowledge graph | NetworkX                                  |

**Retrieval channels** are toggled by four flags (`use_sparse`, `use_image`,
`use_graph`, `use_rerank`). The baseline turns all off; the enhanced arm turns all
on. Each channel produces a ranked list, and the lists are combined with **weighted
Reciprocal Rank Fusion** (by rank, not raw score) so a figure or a graph-reached
chunk can reach the top results instead of being buried under text hits. Reranking
then re-scores the merged set.

**Evaluation is segmented** into three question types so each channel's value is
visible: *text* (gold is a chunk → text retrieval), *visual* (gold is a figure →
image retrieval), and *multihop* (gold is a chunk reachable only through the graph
→ graph retrieval). Generated answers are scored for faithfulness and relevancy.

**Explainability**: every answer carries a provenance record — which evidence was
used, from which channel, its score, whether it was cited — plus the knowledge-graph
paths and the cited figure crops.

## Project layout

```
configs/                  base + baseline + enhanced + ablation YAML
scripts/                  01, 01b, 02, 03, 04, 05 stage entry points
app/demo.py               Gradio UI
dev/                      helper scripts (lint, test, smoke, VLM diagnostic)
report.md                 the 2-page write-up
src/publaynet_mmrag/
  ingest/                 dataset loaders (webdataset, coco)
  preprocess/             region cropping, OCR, chunking
  embed/                  text, image, captioning models
  index/                  embedded Qdrant store + schema
  kg/                     entity/relation extraction, graph build + query
  retrieve/               retriever (RRF fusion) + reranker
  reason/                 prompts, text LLM, vision LLM, generation
  explain/                provenance + visual grounding
  eval/                   question synthesis, retrieval + generation metrics
  pipeline.py             end-to-end assembly
tests/                    pure-logic unit tests (no GPU)
```

## Advanced usage

### Environment and PyTorch (CPU / GPU / CUDA)

`pip install -e .` pulls in `torch`, but the default PyPI wheel may target an older
CUDA. This project needs **CUDA 13+**, so install the matching PyTorch build
**before** `pip install -e .`:

1. Use the official selector at <https://pytorch.org/get-started/locally/> and pick
   your CUDA 13+ build, or install a nightly/preview build if your GPU is very new
   (the test system used `torch 2.12.1+cu132` for a Blackwell RTX 5070).
2. Then run `pip install -e .` — pip sees `torch` already satisfied and won't change
   it. `torchvision` (a dependency, used by the captioner) installs alongside.
3. On a preview/nightly `torch`, if step 2 tries to *change* your `torch` while
   resolving `torchvision`, install torchvision on its own instead:
   `pip install --no-deps torchvision` (this is what the test system used).

CPU-only is not supported for a full run (OCR, embeddings and the LLM are far too
slow), but the unit tests run on CPU with no models. If `torch` does not detect your
GPU after install, reinstall the correct build first.

### Optional extras

A bare `pip install -e .` installs the core only. Add extras as needed:

```bash
pip install -e ".[serve]"      # FastAPI HTTP serving (Stage 5)
pip install -e ".[demo]"       # Gradio UI
pip install -e ".[dev]"        # pytest + ruff
pip install -e ".[quant]"      # bitsandbytes, for a 4-bit larger generator
pip install -e ".[all]"        # serve + demo + dev together
```

### Configuration and flags

`configs/base.yaml` holds every setting; `configs/baseline.yaml` and
`configs/enhanced.yaml` overlay the retrieval flags. Useful knobs:

- `ingest.max_pages` — set small (e.g. `50`) for a quick smoke run.
- `retrieval.{use_sparse,use_image,use_graph,use_rerank}` — toggle channels.
- `retrieval.{rrf_k,text_weight,image_weight,graph_weight}` — fusion weights.
- `eval.{num_questions,num_visual_questions,num_multihop_questions}` — set sizes.
- `eval.judge_sample_size` — how many answers the LLM judge scores per split.
- `generation.vision_generation` — answer with the VLM that reads crops (the demo
  and API set this on; the evaluation leaves it off for comparable numbers).

Stage 4 flags:

- `--variants a,b,c` — which configs to evaluate.
- `--judge-variants a,b` — which of those get the (slower) generation judge;
  default `baseline,enhanced`. Pass `all` to judge every variant.
- `--no-judge` — skip generation scoring entirely (retrieval metrics only, fast).

`scripts/05_serve.py --text-generation` falls back to the text LLM instead of the
crop-reading VLM.

### GPU memory (VRAM) tuning

The pipeline targets a 12 GiB card. Because stages run **one model at a time**, peak
VRAM is set by the single largest model — the generator LLM (Qwen3-4B, ~8 GiB fp16)
and the captioner / visual VLM (Qwen2.5-VL-3B, ~7 GiB). All knobs below are under
`models:` in `configs/base.yaml` (except the eval sizes).

**Less than 12 GiB:**

- `llm_load_in_4bit: true` (needs `".[quant]"`) — drops the generator to ~3 GiB; or
  point `llm_model` at a smaller instruct model.
- Lower `ocr_detector_batch_size` and `ocr_recognition_batch_size` if Stage 1 runs
  out of memory on a dense page.
- Keep `retrieval_device: cpu` and `rerank_device: ""` (also CPU) so the query-time
  retrieval models do not compete with the LLM for VRAM.
- Optionally lower `eval.judge_sample_size` / `eval.caption_sample_size` to shorten
  runs (this reduces load, not peak VRAM).

**16–24 GiB or more:**

- `retrieval_device: cuda` and `rerank_device: cuda` — run the query-time retrieval
  and reranker models on the GPU for a faster Stage 4.
- Raise `ocr_detector_batch_size` / `ocr_recognition_batch_size` for a faster
  Stage 1.
- Use a larger `llm_model` (e.g. an 8B/14B instruct), optionally with
  `llm_load_in_4bit: true` or an `-FP8` checkpoint — see
  [Larger or quantised generators](#larger-or-quantised-generators).

`models.device` controls the build stages (1–3); `models.retrieval_device` controls
the query-time retrieval models, defaulting to CPU to leave VRAM for the LLM.

### Per-component ablations

The single command in [Reproduce the results](#reproduce-the-results) already runs
the four ablations (`abl_rerank`, `abl_hybrid`, `abl_image`, `abl_graph`), each
enabling exactly one enhanced channel on top of the baseline, so the report can
attribute the gain to each component. For a fast retrieval-only sweep add
`--no-judge`.

### Larger or quantised generators

To run a bigger answer model on the same card:

- **bitsandbytes 4-bit**: install `".[quant]"`, set a larger `models.llm_model`
  and `models.llm_load_in_4bit: true`.
- **FP8 checkpoint**: point `models.llm_model` at an `-FP8` id and load with
  `dtype: auto`.

`dev/diagnose_vlm.py` runs a vision model directly on one figure/table crop (no
retrieval) to compare models or input resolutions — handy for inspecting how well a
VLM reads a given figure.

### Scaling to the full dataset

The proof-of-concept uses the `lhoestq/small-publaynet-wds` subset. To use the full
PubLayNet release, change ingestion only:

```yaml
ingest:
  source: coco
  coco_annotations: /path/to/publaynet/train.json
  coco_image_dir: /path/to/publaynet/train
```

No other code changes are needed. At full scale, point the Qdrant client at a server
URL and swap the in-memory graph for an embedded graph database behind the same
`kg.query` interface.

### Resumability and out-of-memory tuning

Stages 1–4 resume automatically: re-run the same command and each skips work already
on disk (region files, indexed points, graph nodes, cached per-variant results), so
a crash mid-run is not fatal. To rebuild a stage, delete its output (the region dir,
the Qdrant dir, the graph file, or the per-variant result files). Ctrl-C and `kill`
shut down gracefully and save progress.

If Stage 1 runs out of GPU memory on a dense page, lower
`models.ocr_detector_batch_size` (and `ocr_recognition_batch_size`) in
`configs/base.yaml`.

### Development

```bash
pip install -e ".[dev]"
dev/lint.sh        # check lint + formatting
dev/lint-fix.sh    # auto-fix and reformat
dev/test.sh        # run the unit tests (no GPU needed)
dev/smoke.sh       # end-to-end pipeline on a single page (needs a GPU)
```

The unit tests cover the framework-free logic (parsing, config, chunking, fusion,
retrieval metrics, KG build/query, the multi-hop filter, metric segmentation) and
need no GPU, downloads or network.

### Version notes

- **Transformers** is pinned to `4.56.1` — the version that works across Surya, the
  Qwen3 text LLM and the Qwen2.5-VL captioner together. It does **not** recognise
  `qwen3-vl`, which is why the captioner/visual generator use Qwen2.5-VL. Do not
  change this pin without re-checking the whole stack.
- **Surya** targets the stable v1.x predictor API (its API changed at v0.20).
- **Qdrant** hybrid search uses the `query_points` + `Prefetch` + `FusionQuery` RRF
  API; the required floor is pinned in `pyproject.toml`.
- **Generation metrics** use a small self-contained LLM judge (the pipeline's own
  model), not an external evaluation library.
