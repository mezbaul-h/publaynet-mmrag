# Multimodal RAG + Knowledge Graph over PubLayNet

**Task & data.** Answer scientific questions over document pages by retrieving and
reasoning across **text and visual** elements, augmenting retrieval with a small
**knowledge graph (KG)**, and producing **explainable** answers — and quantify
whether non-text modality and structured knowledge beat a text-only RAG baseline.
Data: the `lhoestq/small-publaynet-wds` subset of PubLayNet (3,812 pages; 17,366
text chunks; 2,489 figure/table regions).

## 1. Preprocessing & multimodal representation

A **staged, resumable pipeline** writes each stage to disk and frees the GPU
before the next, so models that cannot co-reside on a 12 GiB card load one at a
time. Stage 1 crops every labelled region,
OCRs text regions (Surya), and packs reading-order text into ≤1.2k-char **chunks**
with provenance back to regions. Stage 1b captions a ~500-crop sample of
figures/tables with a VLM (Qwen2.5-VL-3B). Stage 2 embeds chunks (BGE-M3
dense+sparse) and crops (SigLIP2) into an embedded Qdrant store. Stage 3 extracts
entities (GLiNER, zero-shot) and relations (LLM triples) into a NetworkX graph
(67,796 nodes; 13,336 labelled relations). The result is **three aligned views**
of each page — dense+sparse text, SigLIP image vectors, a typed entity graph —
sharing region-level provenance.

## 2. Architecture: baseline vs enhanced

One retriever implements both arms; four flags select channels
(`use_sparse / use_image / use_graph / use_rerank`), so the only variables under
study are modality and structured knowledge.

* **Baseline** — dense text retrieval only.
* **Enhanced** — dense+sparse hybrid text **+** SigLIP2 text→image figure
  retrieval **+** KG neighbour expansion **+** cross-encoder reranking.

**Fusion by Reciprocal Rank Fusion (RRF)** is the central design choice. Each
channel returns a ranked list; fused score = Σ `w/(k+rank)`. Channel raw scores
are on incomparable scales (BGE cosine, SigLIP cosine, graph proximity), so the
original raw-score merge sorted every image and graph hit *below* all text hits —
leaving them inert. RRF fuses by rank, so a top figure or graph-reached bridge
chunk reaches the top-k; the reranker then rescores the fused pool for calibrated
cross-modal order. **KG retrieval (GraphRAG-style):** a query links to its most
*specific* entities (whole-word, dropping ones in >40 chunks as too generic), the
1-hop neighbourhood expands, and chunks mentioning reached entities rank by
proximity — surfacing a bridge chunk vector search misses. **Reasoning /
explainability:** generation (Qwen3-4B) answers only from numbered, cited evidence;
each answer yields a provenance record (modality, channel, score, citation) plus KG
paths and cited crops. Serving/the demo can instead generate with a **VLM that reads
the crops directly** (true visual QA); the evaluation keeps the text generator for
comparable numbers.

## 3. Evaluation methodology

PubLayNet ships no QA, so we synthesise a grounded set (same LLM/corpus for both
arms). The key point is **segmented** evaluation: a single text-gold set cannot
reveal visual or graph value, so we score three typed splits, each with the gold
its channel targets:

* **text** (300 Q) — gold = source chunk → text retrieval.
* **visual** (100 Q) — question authored from a figure/table caption; gold = that
  region → image retrieval.
* **multihop** (41 Q) — names entity A but is answered by a chunk about a
  graph-neighbour B that omits A; gold = that bridge chunk. **Verified** per
  question (baseline dense misses the gold *and* graph expansion reaches it), so
  the split genuinely requires multi-hop retrieval.

Retrieval metrics (Recall@k/MRR/nDCG) run over every question, with exact-id match
on the visual/multihop splits (a same-document sibling is not a figure answer) and
a document fallback only for text. Generation metrics (faithfulness,
answer-relevancy) use an in-process LLM judge per split. We run `baseline`, four
single-channel ablations, and `enhanced`.

## 4. Key results

**Retrieval (Recall@5 / MRR).**

| Split (N)         | Baseline    | Enhanced        | Best single channel |
|-------------------|-------------|-----------------|---------------------|
| text (300)        | 0.97 / 0.85 | **1.00 / 0.93** | rerank 0.98 / 0.92  |
| visual (100)      | 0.00 / 0.00 | **0.56 / 0.54** | image 0.35 / 0.16   |
| multihop (41)     | 0.00 / 0.00 | **0.73 / 0.54** | graph 0.61 / 0.38   |
| **overall (441)** | 0.66 / 0.58 | **0.87 / 0.81** | —                   |

The text-only baseline scores **exactly 0** on visual and multihop — it cannot
retrieve a figure, and the multihop golds are (by construction) outside its top-k.
Modality and structure lift these to 0.56 and 0.73 Recall@5. **Ablations attribute
the gains cleanly:** *all* visual recall is the image channel (every text-side
ablation stays 0); multihop is led by the **graph** (0.61) with sparse (0.41) and
rerank (0.46) also recovering bridge chunks; **rerank** drives the text gain (R@1
0.75→0.87). Fusion is the enabler — RRF surfaces a figure/bridge chunk into the
pool and the reranker promotes it to rank 1 (visual R@1 0.00→**0.53**).

**Generation (faithfulness / answer-relevancy), baseline → enhanced.**

| Split    | Baseline    | Enhanced        |
|----------|-------------|-----------------|
| text     | 0.83 / 0.90 | 0.72 / **0.94** |
| visual   | 0.45 / 0.45 | **0.49 / 0.57** |
| multihop | 0.37 / 0.38 | **0.41 / 0.60** |

Reasoning tracks retrieval: answer-relevancy rises on every split (most on
multihop, 0.38→0.60) once the right evidence is retrieved. Pure-text faithfulness
dips (0.83→0.72) as cross-modal context adds some off-topic evidence — a cost of
fusing modalities without routing.

## 5. Discussion

**Value.** Each non-text channel unlocks a question class the baseline cannot
answer *at all* (visual 0→0.56; multihop 0→0.73) — categorical, not marginal. These
signals existed before but were inert under raw-score merging; **rank-based RRF
fusion** converts them into gains, so the fusion *mechanism* matters as much as the
channels.

**Insights.** (1) A text-gold-only benchmark structurally hides multimodal/KG value
— segmented evaluation is essential. (2) Graph and sparse retrieval are partly
redundant on multi-hop, but the graph captures most (0.61 vs 0.41) by reaching
chunks sharing *no* surface terms with the query. (3) Equal-weight fusion trades a
little text precision for large cross-modal recall (overall R@1 0.51→0.76); a
per-query router is the clear next step.

**Limitations.** Small subset; ~500 captioned crops; a simple self-judge (not
RAGAS); the multihop split is synthetic and verified by construction (it measures
*recoverability* of graph-required evidence); SigLIP/captioner are general-purpose;
no query routing or fine-tuning (out of scope). On *answer* quality, retrieval
cites the correct figure but exact table reading is bounded by **table-VQA, not the
pipeline**: on a dense HER2/EC50 table (PMC5384386, p.2) our 3B VLM reads HER2 max
4,900 correctly but misreads the EC50 max (64.9 vs the true >5,000) — and prompted
on the same table GPT and Claude also miss cells (true HER2 min 3 vs their 4/6).
Salient values are reliable; pinpoint extremes are not, even for frontier models —
the known failure mode of table understanding. The robust fix for tables is
structure OCR (parse cells → compute), not a bigger VLM; for charts, where OCR does
not apply, a sharper/stronger VLM is the lever.

## 6. Reproduction

`pip install -e .`, then run stages `01 → 01b → 02 →
03` and `04_run_eval.py --variants baseline,abl_rerank,abl_hybrid,abl_image,abl_graph,enhanced`
(full commands in the README). Stages are resumable; `dev/test.sh` covers the new
logic (fusion, KG, multi-hop filter, metric segmentation) with no GPU.
