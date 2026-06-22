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

from tqdm import tqdm  # noqa: E402

from publaynet_mmrag.config import Config, load_config  # noqa: E402
from publaynet_mmrag.eval import retrieval_metrics as rm  # noqa: E402
from publaynet_mmrag.eval.build_qa import synthesise_qa  # noqa: E402
from publaynet_mmrag.eval.rag_metrics import evaluate_generation  # noqa: E402
from publaynet_mmrag.pipeline import build_llm, build_system  # noqa: E402
from publaynet_mmrag.reason.llm import LocalLLM  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import Chunk, read_jsonl, write_jsonl  # noqa: E402

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


def _ensure_qa(config: Config, llm: LocalLLM) -> list[dict]:
    """Loads the QA set, synthesising it on first use.

    Args:
        config: The reference configuration (eval + paths settings).
        llm: The shared language model used to write questions.

    Returns:
        The QA rows.
    """
    if os.path.exists(config.paths.qa_path):
        return read_jsonl(config.paths.qa_path)
    chunks = [Chunk.from_dict(r) for r in read_jsonl(config.paths.chunks_path)]
    rows = synthesise_qa(
        chunks=chunks,
        num_questions=config.eval.num_questions,
        llm=llm,
        seed=config.eval.seed,
    )
    write_jsonl(config.paths.qa_path, rows)
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
        A dictionary of retrieval and (optionally) generation metrics.
    """
    system = build_system(config, llm=llm)
    ranks: list = []
    gen_samples: list[dict] = []

    for row in tqdm(qa, desc=f"[{name}] retrieve+gen", unit="q"):
        items, _ = system.retriever.retrieve(row["question"])
        retrieved_ids = [it.chunk_id or it.region_id or "" for it in items]
        retrieved_docs = [it.doc_id for it in items]
        ranks.append(
            rm.gold_rank(
                retrieved_ids,
                retrieved_docs,
                row["gold_chunk_id"],
                row["gold_doc_id"],
            )
        )
        answer = system.generator.generate(row["question"], items, [])
        gen_samples.append(
            {
                "question": row["question"],
                "answer": answer.text,
                "contexts": [it.text for it in items],
            }
        )

    metrics = rm.aggregate(ranks, config.eval.ks)
    if use_judge:
        subset = (
            gen_samples if judge_sample_size <= 0 else gen_samples[:judge_sample_size]
        )
        metrics.update(evaluate_generation(subset, llm=llm, desc=f"[{name}] judge"))
    return metrics


def run(base_path: str, variant_names: list[str], use_judge: bool) -> None:
    """Runs the comparison across all requested variants.

    Args:
        base_path: Path to the base YAML configuration.
        variant_names: Variant config names (without the ``.yaml`` suffix).
        use_judge: Whether to run the generation judge.
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
        qa = _ensure_qa(ref, llm)
        cached = len(results)
        note = f" ({cached} cached)" if cached else ""
        print(f"Evaluating {len(todo)} variant(s) on {len(qa)} questions{note}.")
        for name in tqdm(todo, desc="variants", unit="cfg"):
            variant_path = os.path.join(_CONFIG_DIR, f"{name}.yaml")
            cfg = load_config(base_path, variant_path)
            cfg.mode = name
            metrics = _evaluate_variant(
                cfg, qa, llm, use_judge, ref.eval.judge_sample_size, name
            )
            results[name] = metrics
            with open(_variant_cache_path(ref, name), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
    else:
        print("All requested variants already cached; assembling report.")

    num_questions = (
        len(read_jsonl(ref.paths.qa_path)) if os.path.exists(ref.paths.qa_path) else 0
    )
    ordered = {name: results[name] for name in variant_names if name in results}
    report: dict = {"num_questions": num_questions, "variants": ordered}
    if "baseline" in ordered:
        base_m = ordered["baseline"]
        report["delta_vs_baseline"] = {
            name: {
                k: round(ordered[name][k] - base_m.get(k, 0.0), 4)
                for k in ordered[name]
            }
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
    args = parser.parse_args()
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    from publaynet_mmrag.shutdown import graceful_shutdown

    with graceful_shutdown(message="Stage 4 interrupted."):
        run(args.config, variant_names, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
