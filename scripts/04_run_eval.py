#!/usr/bin/env python
"""Stage 4: evaluate and compare configurations (baseline, enhanced, ablations).

Runs one or more variants over a shared synthetic QA set. Retrieval metrics
(Recall@k, MRR, nDCG) are computed for *every* question (cheap, no LLM); the
generation judge (faithfulness, answer-relevancy) runs over a sampled subset
(`eval.judge_sample_size`) since each judged answer costs two extra LLM calls.

A single language model is loaded once and shared across QA synthesis, every
variant and the judge. Per-variant metrics plus deltas vs the baseline are
written to the results directory.

Examples:
    # Default: baseline vs enhanced.
    python scripts/04_run_eval.py

    # Full per-component ablation.
    python scripts/04_run_eval.py --variants \\
        baseline,abl_rerank,abl_hybrid,abl_image,abl_graph,enhanced

    # Fast retrieval-only sweep (no LLM judge).
    python scripts/04_run_eval.py --variants baseline,enhanced --no-judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import glob  # noqa: E402
from collections import defaultdict  # noqa: E402

from tqdm import tqdm  # noqa: E402

from publaynet_mmrag.config import Config, load_config  # noqa: E402
from publaynet_mmrag.eval import retrieval_metrics as rm  # noqa: E402
from publaynet_mmrag.eval.build_qa import (  # noqa: E402
    normalise_row,
    synthesise_multihop_qa,
    synthesise_text_qa,
    synthesise_visual_qa,
)
from publaynet_mmrag.eval import rag_metrics as rmg  # noqa: E402
from publaynet_mmrag.kg import query as kg_query  # noqa: E402
from publaynet_mmrag.kg.build import load_graph  # noqa: E402
from publaynet_mmrag.pipeline import build_llm, build_system  # noqa: E402
from publaynet_mmrag.reason.llm import LocalLLM  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import (  # noqa: E402
    Category,
    Chunk,
    Region,
    read_jsonl,
    write_jsonl,
)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "configs")


def _variant_cache_path(ref, name: str) -> str:
    """Returns the per-variant result cache path.

    Args:
        ref: The reference config (for the results directory).
        name: Variant name.

    Returns:
        Path to the variant's cached metrics JSON.
    """
    return os.path.join(ref.paths.results_dir, f"_variant_{name}.json")


def _load_captioned_regions(config: Config) -> list[Region]:
    """Loads captioned visual regions for the visual QA split.

    Args:
        config: The active configuration (region directory).

    Returns:
        Visual regions (figure/table) that carry a non-empty caption.
    """
    regions: list[Region] = []
    for path in sorted(glob.glob(os.path.join(config.paths.regions_dir, "*.jsonl"))):
        for row in read_jsonl(path):
            region = Region.from_dict(row)
            if region.category in Category.visual() and region.caption:
                regions.append(region)
    return regions


def _ensure_qa(base_path: str, config: Config, llm: LocalLLM) -> list[dict]:
    """Loads the typed QA set, synthesising any missing split on first use.

    Existing rows are preserved (legacy rows are upgraded to ``text``); only the
    splits not yet present are generated and appended, so re-runs do not recompute
    questions and adding a new split is incremental.

    Args:
        base_path: Path to the base YAML (to build the baseline retriever used to
            verify multi-hop questions).
        config: The reference configuration (eval + paths settings).
        llm: The shared language model used to write questions.

    Returns:
        The combined QA rows across the text, visual and multihop splits.
    """
    rows: list[dict] = []
    if os.path.exists(config.paths.qa_path):
        rows = [normalise_row(r) for r in read_jsonl(config.paths.qa_path)]
    present = {r["qtype"] for r in rows}

    chunks: list[Chunk] | None = None
    new: list[dict] = []

    if "text" not in present and config.eval.num_questions > 0:
        chunks = [Chunk.from_dict(r) for r in read_jsonl(config.paths.chunks_path)]
        new += synthesise_text_qa(
            chunks, config.eval.num_questions, llm, seed=config.eval.seed
        )

    if "visual" not in present and config.eval.num_visual_questions > 0:
        regions = _load_captioned_regions(config)
        if regions:
            new += synthesise_visual_qa(
                regions, config.eval.num_visual_questions, llm, seed=config.eval.seed
            )
        else:
            print(
                "No captioned visual regions found; skipping the visual split. "
                "Run scripts/01b_caption_regions.py first."
            )

    if "multihop" not in present and config.eval.num_multihop_questions > 0:
        if os.path.exists(config.paths.kg_path):
            if chunks is None:
                chunks = [
                    Chunk.from_dict(r) for r in read_jsonl(config.paths.chunks_path)
                ]
            new += _synthesise_multihop(base_path, config, chunks, llm)
        else:
            print("No knowledge graph found; skipping the multihop split.")

    if new:
        rows += new
        write_jsonl(config.paths.qa_path, rows)
    return rows


def _synthesise_multihop(
    base_path: str, config: Config, chunks: list[Chunk], llm: LocalLLM
) -> list[dict]:
    """Builds the multihop split, verifying each question needs the graph.

    Args:
        base_path: Path to the base YAML configuration.
        config: The reference configuration.
        chunks: All corpus chunks.
        llm: The shared language model.

    Returns:
        Verified multihop QA rows.
    """
    graph = load_graph(config.paths.kg_path)
    baseline_cfg = load_config(base_path, os.path.join(_CONFIG_DIR, "baseline.yaml"))
    baseline = build_system(baseline_cfg, llm=llm).retriever
    candidate_k = config.retrieval.candidate_k
    hops = config.retrieval.graph_hops

    def dense_miss_fn(question: str, gold_chunk_id: str) -> bool:
        items, _ = baseline.retrieve(question)
        return gold_chunk_id not in [it.chunk_id for it in items]

    def graph_reach_fn(question: str, gold_chunk_id: str) -> bool:
        expansion = kg_query.expand(graph, question, hops=hops)
        return gold_chunk_id in expansion.chunk_ids[:candidate_k]

    rows = synthesise_multihop_qa(
        graph=graph,
        chunks=chunks,
        num_questions=config.eval.num_multihop_questions,
        llm=llm,
        dense_miss_fn=dense_miss_fn,
        graph_reach_fn=graph_reach_fn,
        seed=config.eval.seed,
    )
    print(
        f"Multihop split: {len(rows)} verified questions "
        f"(target {config.eval.num_multihop_questions})."
    )
    return rows


def _evaluate_variant(
    config: Config,
    qa: list[dict],
    llm: LocalLLM,
    use_judge: bool,
    judge_sample_size: int,
    name: str,
) -> dict:
    """Runs one variant over the QA set and scores it.

    Args:
        config: The variant configuration.
        qa: The shared QA set.
        llm: The shared language model.
        use_judge: Whether to run the generation judge.
        judge_sample_size: Number of answers to judge (0 = all).
        name: Variant name, used for progress labels.

    Returns:
        A nested dict with ``overall`` metrics and a ``by_type`` block giving the
        same metrics for each question type (text / visual / multihop).
    """
    system = build_system(config, llm=llm)
    ranks_by_type: dict[str, list] = defaultdict(list)
    # Retrieved evidence kept per type so generation runs only on judged samples.
    records_by_type: dict[str, list[tuple[str, list]]] = defaultdict(list)

    for row in tqdm(qa, desc=f"[{name}] retrieve", unit="q"):
        items, _ = system.retriever.retrieve(row["question"])
        retrieved_ids = [it.chunk_id or it.region_id or "" for it in items]
        retrieved_docs = [it.doc_id for it in items]
        qtype = row.get("qtype", "text")
        ranks_by_type[qtype].append(
            rm.gold_rank(
                retrieved_ids,
                retrieved_docs,
                row["gold_id"],
                row["gold_doc_id"],
                allow_doc_fallback=(qtype == "text"),
            )
        )
        records_by_type[qtype].append((row["question"], items))

    # Generate answers for the judged subset of each type once, score them once,
    # then pool the per-sample scores for the overall figure (no re-generation).
    scored_by_type: dict[str, list[dict]] = {}
    for qtype, records in records_by_type.items():
        if not use_judge:
            scored_by_type[qtype] = []
            continue
        subset = records if judge_sample_size <= 0 else records[:judge_sample_size]
        gens = []
        for question, items in tqdm(subset, desc=f"[{name}/{qtype}] gen", unit="ans"):
            answer = system.generator.generate(question, items, [])
            gens.append(
                {
                    "question": question,
                    "answer": answer.text,
                    "contexts": [it.text for it in items],
                }
            )
        scored_by_type[qtype] = rmg.score_samples(
            gens, llm=llm, desc=f"[{name}/{qtype}] judge"
        )

    def _metrics(ranks: list, scored: list[dict]) -> dict:
        metrics = rm.aggregate(ranks, config.eval.ks)
        metrics.update(rmg.aggregate_scores(scored))
        return metrics

    all_ranks = [r for rs in ranks_by_type.values() for r in rs]
    all_scored = [s for ss in scored_by_type.values() for s in ss]
    return {
        "overall": _metrics(all_ranks, all_scored),
        "by_type": {
            qtype: _metrics(ranks_by_type[qtype], scored_by_type.get(qtype, []))
            for qtype in sorted(ranks_by_type)
        },
    }


def _delta_block(variant: dict, base: dict) -> dict:
    """Computes per-metric deltas for one metrics block.

    Args:
        variant: The variant's metrics block.
        base: The baseline's metrics block.

    Returns:
        ``variant - baseline`` for each shared metric, rounded.
    """
    return {k: round(variant[k] - base.get(k, 0.0), 4) for k in variant}


def _delta(variant: dict, base: dict) -> dict:
    """Computes deltas vs baseline for both overall and per-type metrics.

    Args:
        variant: A variant's nested ``{overall, by_type}`` metrics.
        base: The baseline's nested metrics.

    Returns:
        A nested delta record mirroring the metrics structure.
    """
    out: dict = {"overall": _delta_block(variant["overall"], base["overall"])}
    out["by_type"] = {
        qtype: _delta_block(metrics, base["by_type"][qtype])
        for qtype, metrics in variant.get("by_type", {}).items()
        if qtype in base.get("by_type", {})
    }
    return out


def run(
    base_path: str,
    variant_names: list[str],
    use_judge: bool,
    judge_variants: set[str] | None = None,
) -> None:
    """Runs the comparison across all requested variants.

    Args:
        base_path: Path to the base YAML configuration.
        variant_names: Variant config names (without the ``.yaml`` suffix).
        use_judge: Whether to run the generation judge at all.
        judge_variants: If given, only these variants are judged (the others get
            the cheap retrieval metrics only). Generation is far slower than
            retrieval, so judging just the two arms (baseline, enhanced) -- which
            carry the end-to-end reasoning story -- while leaving the retrieval
            ablations judge-free keeps the run tractable.
    """
    # A reference config (any variant) supplies shared eval/model/path settings.
    ref = load_config(base_path, os.path.join(_CONFIG_DIR, "enhanced.yaml"))
    use_judge = use_judge and ref.eval.use_llm_judge
    os.makedirs(ref.paths.results_dir, exist_ok=True)

    import time

    start = time.perf_counter()

    # Load any already-completed variants and decide what is left to run, so a
    # re-run after a crash/interrupt skips variants already finished. To force a
    # fresh evaluation, delete the _variant_*.json files (or the results dir).
    results: dict[str, dict] = {}
    todo: list[str] = []
    for name in variant_names:
        cache_path = _variant_cache_path(ref, name)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                results[name] = json.load(handle)
        else:
            todo.append(name)

    if todo:
        llm = build_llm(ref)
        qa = _ensure_qa(base_path, ref, llm)
        cached = len(results)
        note = f" ({cached} cached)" if cached else ""
        print(f"Evaluating {len(todo)} variant(s) on {len(qa)} questions{note}.")
        for name in tqdm(todo, desc="variants", unit="cfg"):
            variant_path = os.path.join(_CONFIG_DIR, f"{name}.yaml")
            cfg = load_config(base_path, variant_path)
            cfg.mode = name
            judge_this = use_judge and (
                judge_variants is None or name in judge_variants
            )
            metrics = _evaluate_variant(
                cfg, qa, llm, judge_this, ref.eval.judge_sample_size, name
            )
            results[name] = metrics
            with open(_variant_cache_path(ref, name), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
    else:
        print("All requested variants already cached; assembling report.")

    qa_rows = read_jsonl(ref.paths.qa_path) if os.path.exists(ref.paths.qa_path) else []
    num_questions: dict[str, int] = {"total": len(qa_rows)}
    for row in qa_rows:
        qtype = row.get("qtype", "text")
        num_questions[qtype] = num_questions.get(qtype, 0) + 1

    ordered = {name: results[name] for name in variant_names if name in results}
    report: dict = {"num_questions": num_questions, "variants": ordered}
    if "baseline" in ordered:
        report["delta_vs_baseline"] = {
            name: _delta(ordered[name], ordered["baseline"])
            for name in ordered
            if name != "baseline"
        }

    out_path = os.path.join(ref.paths.results_dir, "comparison.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nComparison written to {out_path}")
    print(f"Stage 4 finished in {format_duration(time.perf_counter() - start)}.")


def main() -> None:
    """Parses arguments and runs Stage 4."""
    parser = argparse.ArgumentParser(description="Stage 4: evaluate and compare.")
    parser.add_argument(
        "--config",
        default=os.path.join(_CONFIG_DIR, "base.yaml"),
        help="Path to the base YAML configuration.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,enhanced",
        help="Comma-separated variant config names (files in configs/).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (retrieval metrics only; much faster).",
    )
    parser.add_argument(
        "--judge-variants",
        default="baseline,enhanced",
        help="Only judge these variants (others get retrieval metrics only). "
        "Generation is much slower than retrieval, so the ablations -- which "
        "exist to attribute retrieval -- are left judge-free by default. "
        "Pass 'all' to judge every variant.",
    )
    args = parser.parse_args()
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    judge_variants: set[str] | None = None
    if args.judge_variants.strip().lower() != "all":
        judge_variants = {
            v.strip() for v in args.judge_variants.split(",") if v.strip()
        }
    from publaynet_mmrag.shutdown import graceful_shutdown

    with graceful_shutdown(message="Stage 4 interrupted."):
        run(
            args.config,
            variant_names,
            use_judge=not args.no_judge,
            judge_variants=judge_variants,
        )


if __name__ == "__main__":
    main()
