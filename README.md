# PubLayNet Multimodal RAG + Knowledge Graph

A staged retrieval-augmented generation pipeline over the
[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) document-layout dataset.
It answers scientific questions by reasoning over both text and visual page
elements, augments retrieval with a small knowledge graph, and reports a
quantitative comparison between a **text-only baseline** and an **enhanced
multimodal + graph** pipeline.

Everything runs in-process. There is **no external inference server, no Docker,
and no system binary to install** -- only `pip` packages, whose model weights are
fetched from the Hugging Face Hub on first use. Generation runs locally through
Transformers.

---

## 1. Why a staged pipeline

The binding constraint on a 12 GiB GPU is that the models cannot all be resident
at once. The pipeline is therefore split into stages; each loads only the models
it needs, writes its output to disk, and frees the GPU before the next stage
starts. This also makes every stage independently resumable, which matters when
scaling to the full ~96 GB dataset.

```
Stage 1  ingest + crop + OCR (+ optional caption)   -> regions, chunks, crops
Stage 2  embed text (BGE-M3) + images (SigLIP2)      -> Qdrant index
Stage 3  entities (GLiNER) + relations (LLM)         -> knowledge graph
Stage 4  synthesise QA, run both arms, score         -> comparison.json
Stage 5  serve (optional) / demo (optional)
```

| Component        | Model / tool                              |
|------------------|-------------------------------------------|
| OCR              | Surya (v1.x), pip + HF weights            |
| Text embedding   | BAAI/bge-m3 (dense + learned sparse)      |
| Image embedding  | google/siglip2-base-patch16-224           |
| Reranking        | BAAI/bge-reranker-v2-m3                   |
| Entities (NER)   | urchade/gliner_medium-v2.1                |
| Captioning       | Qwen/Qwen3-VL-4B-Instruct (optional)      |
| Vector store     | Qdrant (embedded, on-disk; no server)     |
| Knowledge graph  | NetworkX MultiDiGraph                     |
| Generation       | Qwen3-4B-Instruct-2507, in-process (Transformers) |

Every model is loaded in-process via `pip` packages; nothing else is required on
the host.

## 2. The baseline-vs-enhanced comparison

Both arms share the same OCR, chunks, LLM and evaluation set; they differ only
in the retrieval channels, set by four flags:

| Flag          | Baseline | Enhanced | What it adds                          |
|---------------|----------|----------|---------------------------------------|
| `use_sparse`  | off      | on       | dense+sparse hybrid text retrieval    |
| `use_image`   | off      | on       | SigLIP2 figure/table retrieval        |
| `use_graph`   | off      | on       | knowledge-graph neighbour expansion   |
| `use_rerank`  | off      | on       | cross-encoder reranking               |

Toggle them individually in `configs/enhanced.yaml` to ablate each component.

## 3. Setup

The project installs with either a pip virtual environment or a conda
environment. `pyproject.toml` is the single source of truth for dependencies;
`requirements.txt` and `environment.yml` mirror it.

### 3.1 pip + venv

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

### 3.2 conda

Conda provides the interpreter and the environment boundary; the project is
installed via pip inside it (several dependencies are PyPI-only). Run from the
repository root so the editable install resolves:

```bash
conda env create -f environment.yml
conda activate publaynet-mmrag
# extras: pip install -e ".[all]"   (or edit the pip line in environment.yml)
```

To have conda manage CUDA-enabled PyTorch instead of pip, uncomment the
`pytorch` / `pytorch-cuda` lines and the `pytorch` / `nvidia` channels in
`environment.yml` and set the CUDA version to match your driver.

> **⚠ PyTorch / CUDA — read before installing.**
> The install above pulls in `torch`. On Linux, the default PyPI
> wheel **bundles CUDA**, so **pip installs a GPU-enabled build by default**.
> That build is what you want on a CUDA machine, but the bundled CUDA version
> may not match your driver, and Windows / macOS / CPU-only / ROCm setups need a
> different wheel. If the default build does not detect your GPU, or you need a
> specific CUDA / CPU / ROCm build, install the matching `torch` **first** using
> the official selector at <https://pytorch.org/get-started/locally/>, then
> reinstall.

Optional extras, installed only if you need them. A bare `pip install -e .`
installs the **core dependencies only** -- none of the extras below. There is no
built-in pip keyword for "everything", so an `all` extra is provided that bundles
`serve`, `demo` and `dev` (but not `quant`, which is hardware-sensitive):

```bash
pip install -e ".[all]"      # serve + demo + dev in one go
pip install -e ".[serve]"    # FastAPI HTTP serving (Stage 5)
pip install -e ".[demo]"     # Gradio UI
pip install -e ".[dev]"      # pytest + ruff
pip install -e ".[quant]"    # bitsandbytes, for a 4-bit larger generator
pip install -e ".[all,quant]"  # truly everything, if you want 4-bit too
```

## 4. The generator

The default generator is **Qwen3-4B-Instruct-2507 in FP16**, loaded in-process. It
needs only the core install and fits a 12 GiB GPU alongside the retrieval
models, which run on the CPU at query time (`models.retrieval_device: cpu`) so
they do not compete with the LLM for VRAM. The stack pins Transformers to
exactly `4.56.1` (see `pyproject.toml`) -- the only version observed to work
across Surya, the Qwen3 LLM and the Qwen3-VL captioner together. Do not upgrade
or downgrade it.

To run a larger model on the same card there are two routes:

- **bitsandbytes 4-bit** (works on any recent CUDA GPU): install the `quant`
  extra and set a larger `llm_model` with `llm_load_in_4bit: true`.
- **FP8 checkpoint** (best on Ada/Hopper GPUs, compute capability >= 8.9; on
  Ampere it loads weight-only without the compute speed-up): point `llm_model`
  at an `-FP8` checkpoint and leave `llm_load_in_4bit: false`. No extra package
  is needed, but load with `dtype: auto` so the weights stay in fp8.

```yaml
models:
  llm_model: <larger-qwen3-instruct>     # e.g. an 8B/14B dense instruct
  llm_load_in_4bit: true                 # bitsandbytes route; omit for an -FP8 id
```

## 5. Running the pipeline

Run the stages in order. A quick smoke test: set `ingest.max_pages: 50` in
`configs/base.yaml` first.

```bash
# Stage 1 - ingest, crop, OCR. Add --caption-figures to caption visual regions.
python scripts/01_ingest_preprocess.py --mode enhanced

# Stage 2 - build the Qdrant index (text + images).
python scripts/02_build_index.py --mode enhanced

# Stage 3 - build the knowledge graph.
python scripts/03_build_kg.py --mode enhanced

# Stage 4 - synthesise QA and compare both arms.
python scripts/04_run_eval.py

# Stage 5 - serve or demo (optional extras required).
python scripts/05_serve.py --mode enhanced          # needs [serve]; :8000
python app/demo.py                                  # needs [demo]
```

Stage 4 writes `data/results/comparison.json`. By default it compares
`baseline` vs `enhanced`; pass `--variants` to attribute the gain to individual
components, each config enabling one channel on top of the baseline:

```bash
# Per-component ablation in one run.
python scripts/04_run_eval.py --variants \
    baseline,abl_rerank,abl_hybrid,abl_image,abl_graph,enhanced

# Fast retrieval-only sweep (skip the LLM judge).
python scripts/04_run_eval.py --variants baseline,abl_rerank,abl_hybrid,abl_image,abl_graph,enhanced --no-judge
```

The report then holds metrics for every variant plus `delta_vs_baseline`.
Eval-cost controls: `eval.num_questions` sizes the QA set (retrieval metrics use
all of them and call no LLM); `eval.judge_sample_size` caps how many answers the
LLM judge scores (each costs two extra LLM calls; `0` = all); `--no-judge` (or
`eval.use_llm_judge: false`) skips generation metrics entirely, which makes
ablation sweeps many times faster.

**Progress and downloads.** Model weights are fetched from the Hugging Face Hub
on first use, lazily per stage, each with a download progress bar; they cache to
`~/.cache/huggingface` so later runs are quiet. Every stage shows a `tqdm`
progress bar (Stage 1 over pages, Stage 2 over text/image batches, Stage 3 over
chunks, Stage 4 over questions and judged answers), so a run never looks frozen
and bars carry an ETA wherever the total is known. The exception is Stage 1
while **streaming** (the default): the page total is not known up front, so its
bar shows a running count and rate rather than an ETA -- set
`ingest.streaming: false` in `configs/base.yaml` to download the four shards
first (with a download bar) and get a full ETA over the subset (fine at 1.22 GB).
Surya's own per-page detection/recognition bars are hidden by default so they do
not scroll the Stage 1 bar out of view; pass `--verbose-ocr` to Stage 1 to show
them.

## 6. Scaling to the full dataset

The proof-of-concept runs on the `lhoestq/small-publaynet-wds` subset. To move
to the full release, change ingestion only:

```yaml
ingest:
  source: coco
  coco_annotations: /path/to/publaynet/train.json
  coco_image_dir: /path/to/publaynet/train
```

No other code changes are needed: the ingestion interface yields the same
`PageSample` contract, and the Qdrant client switches from embedded to a server
URL by changing one constructor argument. At full scale, swap the in-memory
NetworkX graph for an embedded graph database (e.g. Kuzu) behind the same
`kg.query` interface.

## 7. Development

Install the dev tools (`ruff`, `pytest`) and use the helper scripts in `dev/`:

```bash
pip install -e ".[dev]"

dev/lint.sh        # check lint + formatting (no changes)
dev/lint-fix.sh    # auto-fix lint and reformat in place
dev/test.sh        # run the unit-test suite (extra args pass through to pytest)
dev/smoke.sh       # end-to-end pipeline smoke test on a single page
```

The unit tests cover the framework-free logic (parsing, config composition,
chunking, retrieval metrics, KG build, NER windowing, the duration formatter and
the dependency guards) and need no GPU, model downloads or network access.

`dev/smoke.sh` runs the full pipeline -- stages 1 through 4 in enhanced mode --
on a single page, to verify everything wires together end to end. (Stage 5,
serving, is an optional feature rather than part of the pipeline, so it is not
included.) It writes all artifacts under `data/smoke/` so a real run's `data/`
is untouched, builds its config from `configs/base.yaml` on the fly (no
duplication), and uses a tiny QA set. It is a wiring check, not a quality check:
with one page the metrics are meaningless. It needs a GPU and downloads the
models on first use.

## 8. Version notes

* **Surya** changed its API at v0.20 (a VLM inference manager replaced the
  foundation predictor and the output schema changed). This code targets the
  stable v1.x predictor API and pins Surya to the v1.x line in `pyproject.toml`.
  Verify against the
  installed version if you bump it.
* **Qdrant** hybrid search uses the `query_points` + `Prefetch` + `FusionQuery`
  RRF API; the required floor is pinned in `pyproject.toml`.
* **Generation metrics** use a small self-contained LLM judge (the pipeline's
  own model), not an external evaluation library.

## 9. Project layout

```
configs/                  base + baseline + enhanced YAML
scripts/                  01..05 stage entry points
app/demo.py               Gradio UI (optional)
src/publaynet_mmrag/
  ingest/                 source-agnostic loaders (webdataset, coco)
  preprocess/             region cropping, Surya OCR, chunking
  embed/                  text (BGE-M3), image (SigLIP2), captioning
  index/                  embedded Qdrant store + schema
  kg/                     entity/relation extraction, graph build + query
  retrieve/               unified baseline/enhanced retriever + reranker
  reason/                 prompts, in-process LLM, generation
  explain/                provenance + visual grounding
  eval/                   QA synthesis, retrieval metrics, LLM-judge metrics
  pipeline.py             end-to-end system assembly
tests/                    pure-logic unit tests
```
