"""Unit tests for the framework-free pipeline logic.

These cover the pure components that do not need a GPU, model downloads or
network access: key parsing, configuration composition, chunking, retrieval
metrics, and knowledge-graph construction.
"""

from __future__ import annotations

import os
import tempfile

from publaynet_mmrag.config import _deep_merge, load_config
from publaynet_mmrag.eval import retrieval_metrics as rm
from publaynet_mmrag.ingest.base import parse_key
from publaynet_mmrag.kg.build import KnowledgeGraphBuilder, load_graph
from publaynet_mmrag.kg.extract import Entity, Triple
from publaynet_mmrag.preprocess.chunk import chunk_page_regions
from publaynet_mmrag.types import BBox, Category, Chunk, Region


def test_parse_key_splits_doc_and_page():
    """A standard key splits into document id and integer page index."""
    doc, page = parse_key("PMC4991227_00003")
    assert doc == "PMC4991227"
    assert page == 3


def test_parse_key_handles_unexpected_format():
    """A key without the page suffix falls back to page zero."""
    doc, page = parse_key("oddkey")
    assert doc == "oddkey"
    assert page == 0


def test_bbox_xyxy_conversion():
    """COCO x/y/w/h converts to integer corner coordinates."""
    box = BBox.from_coco([10.0, 20.0, 30.0, 40.0])
    assert box.xyxy() == (10, 20, 40, 60)


def test_deep_merge_overlays_nested():
    """Overlay values win and nested dicts merge rather than replace."""
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    overlay = {"nested": {"y": 9}, "b": 3}
    merged = _deep_merge(base, overlay)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 9}, "b": 3}
    # Inputs are untouched.
    assert base["nested"]["y"] == 2


def test_config_composition_disables_enhanced_for_baseline():
    """The baseline variant turns every enhanced retrieval flag off."""
    root = os.path.join(os.path.dirname(__file__), os.pardir, "configs")
    baseline = load_config(
        os.path.join(root, "base.yaml"), os.path.join(root, "baseline.yaml")
    )
    enhanced = load_config(
        os.path.join(root, "base.yaml"), os.path.join(root, "enhanced.yaml")
    )
    assert not any(
        [
            baseline.retrieval.use_sparse,
            baseline.retrieval.use_image,
            baseline.retrieval.use_graph,
            baseline.retrieval.use_rerank,
        ]
    )
    assert all(
        [
            enhanced.retrieval.use_sparse,
            enhanced.retrieval.use_image,
            enhanced.retrieval.use_graph,
            enhanced.retrieval.use_rerank,
        ]
    )


def _region(doc: str, page: int, ann_id: int, text: str, order: int) -> Region:
    """Builds a text region for chunking tests."""
    return Region(
        region_id=f"{doc}:{page}:{ann_id}",
        doc_id=doc,
        page_key=f"{doc}_{page:05d}",
        page_index=page,
        category=Category.TEXT,
        bbox=BBox(0, order * 10, 100, 10),
        reading_order=order,
        text=text,
    )


def test_chunking_packs_and_overlaps():
    """Regions pack into chunks under the size cap with provenance retained."""
    regions = [_region("DOC", 0, i, f"sentence number {i} " * 10, i) for i in range(5)]
    chunks = chunk_page_regions(regions, max_chars=300, overlap_chars=30, min_chars=5)
    assert len(chunks) > 1
    assert all(c.doc_id == "DOC" for c in chunks)
    assert all(c.region_ids for c in chunks)
    assert all(len(c.text) <= 300 + 30 for c in chunks)


def test_chunking_drops_short_regions():
    """Regions below the minimum length are excluded."""
    regions = [_region("DOC", 0, 0, "hi", 0)]
    assert chunk_page_regions(regions, 300, 30, min_chars=40) == []


def test_ablation_configs_isolate_one_component():
    """Each ablation config enables exactly one enhanced channel."""
    root = os.path.join(os.path.dirname(__file__), os.pardir, "configs")
    expected = {
        "abl_rerank": "use_rerank",
        "abl_hybrid": "use_sparse",
        "abl_image": "use_image",
        "abl_graph": "use_graph",
    }
    flags = ["use_sparse", "use_image", "use_graph", "use_rerank"]
    for name, on_flag in expected.items():
        cfg = load_config(
            os.path.join(root, "base.yaml"), os.path.join(root, f"{name}.yaml")
        )
        on = [f for f in flags if getattr(cfg.retrieval, f)]
        assert on == [on_flag], f"{name} should enable only {on_flag}, got {on}"


def test_webdataset_expected_len_none_when_streaming():
    """A streaming source reports no up-front length (so bars show a count)."""
    from publaynet_mmrag.ingest.webdataset_source import WebDatasetSource

    source = WebDatasetSource(shard_urls=["x.tar"], streaming=True)
    assert source.expected_len() is None


def test_ner_windowing_splits_long_text():
    """Long passages split into overlapping windows; short text stays whole."""
    from publaynet_mmrag.kg.extract import EntityExtractor

    extractor = EntityExtractor(
        model_name="x", labels=["method"], window_words=50, overlap_words=10
    )
    short = " ".join(["word"] * 30)
    assert extractor._windows(short) == [short]

    long_text = " ".join(f"w{i}" for i in range(130))
    windows = extractor._windows(long_text)
    assert len(windows) > 1
    # Every window respects the word budget.
    assert all(len(w.split()) <= 50 for w in windows)
    # Coverage: the last word appears in the final window.
    assert "w129" in windows[-1]


def test_bitsandbytes_guard_raises_when_missing():
    """The 4-bit guard fails fast with install guidance when bnb is absent."""
    import importlib.util

    import pytest

    from publaynet_mmrag.reason.llm import _ensure_bitsandbytes

    if importlib.util.find_spec("bitsandbytes") is not None:
        pytest.skip("bitsandbytes is installed; guard not exercised")
    with pytest.raises(ImportError, match=r"quant"):
        _ensure_bitsandbytes()


def test_format_duration_human_readable():
    """Durations render in seconds / minutes / hours as appropriate."""
    from publaynet_mmrag.timing import format_duration

    assert format_duration(7.4) == "7.4s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3723) == "1h 02m 03s"
    assert format_duration(59.95) == "60.0s" or format_duration(59.95).endswith("s")


def test_kg_builder_resume_skips_processed_chunks():
    """from_graph round-trips the graph and reports processed chunk ids."""
    from publaynet_mmrag.kg.build import KnowledgeGraphBuilder
    from publaynet_mmrag.kg.extract import Entity

    builder = KnowledgeGraphBuilder(cooccurrence=True)
    chunk = Chunk(
        chunk_id="DOC:0:c0", doc_id="DOC", page_index=0, text="t", region_ids=["r"]
    )
    builder.add_chunk(chunk, [Entity(text="BERT", label="model", score=0.9)])
    assert "DOC:0:c0" in builder.processed_chunk_ids()

    # Re-wrapping the same graph preserves processed ids and the entity index.
    resumed = KnowledgeGraphBuilder.from_graph(builder.graph, cooccurrence=True)
    assert resumed.processed_chunk_ids() == {"DOC:0:c0"}
    assert resumed._name_index.get("bert") is not None


def test_streaming_defaults_to_false():
    """Streaming is off by default (download up front, get an ETA)."""
    from publaynet_mmrag.config import IngestConfig

    assert IngestConfig().streaming is False


def test_clamp_box_handles_out_of_bounds_and_degenerate():
    """Boxes past the edge are clamped; zero-area or outside boxes return None."""
    from publaynet_mmrag.preprocess.regions import _clamp_box
    from publaynet_mmrag.types import BBox

    assert _clamp_box(BBox.from_coco([90, 90, 50, 50]), 100, 100) == (90, 90, 100, 100)
    assert _clamp_box(BBox.from_coco([200, 200, 10, 10]), 100, 100) is None
    assert _clamp_box(BBox.from_coco([10, 10, 0, 5]), 100, 100) is None
    assert _clamp_box(BBox.from_coco([10, 10, 20, 20]), 100, 100) == (10, 10, 30, 30)


def test_silence_stderr_suppresses_and_restores():
    """stderr is swapped inside the context and restored on exit."""
    import sys

    from publaynet_mmrag.quiet import silence_stderr

    original = sys.stderr
    with silence_stderr():
        assert sys.stderr is not original
        sys.stderr.write("this should be discarded")  # must not raise
    assert sys.stderr is original


def test_extract_regions_skips_tiny_visual_crops(tmp_path):
    """A 1px-tall figure crop is skipped; a normal one is saved."""
    from types import SimpleNamespace

    from PIL import Image

    from publaynet_mmrag.preprocess.regions import extract_regions

    page = SimpleNamespace(
        image=Image.new("RGB", (100, 100), (255, 255, 255)),
        doc_id="DOC",
        page_index=0,
        key="DOC_0",
        annotations=[
            {"category_id": 5, "bbox": [10, 10, 40, 40], "id": 1},  # normal figure
            {"category_id": 5, "bbox": [10, 10, 40, 1], "id": 2},  # 1px tall -> skip
        ],
    )
    regions = extract_regions(page, str(tmp_path))
    by_id = {r.region_id.split(":")[-1]: r for r in regions}
    assert by_id["1"].crop_path is not None
    assert by_id["2"].crop_path is None


def test_graceful_shutdown_runs_cleanup_and_exits():
    """Ctrl-C runs the cleanup, exits 130, and restores the SIGTERM handler."""
    import signal

    import pytest

    from publaynet_mmrag.shutdown import graceful_shutdown

    called = []
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exc_info:
        with graceful_shutdown(on_interrupt=lambda: called.append(True), message="x"):
            raise KeyboardInterrupt
    assert exc_info.value.code == 130
    assert called == [True]
    assert signal.getsignal(signal.SIGTERM) == original


def test_graceful_shutdown_normal_path_skips_cleanup():
    """On normal completion the cleanup does not run and the handler restores."""
    import signal

    from publaynet_mmrag.shutdown import graceful_shutdown

    called = []
    original = signal.getsignal(signal.SIGTERM)
    with graceful_shutdown(on_interrupt=lambda: called.append(True)):
        pass
    assert called == []
    assert signal.getsignal(signal.SIGTERM) == original


def test_retrieval_metrics_basic():
    """Recall, MRR and nDCG follow the gold rank as expected."""
    rank = rm.gold_rank(["a", "b", "c"], ["d1", "d2", "d3"], "b", "d2")
    assert rank == 2
    assert rm.recall_at_k(rank, 1) == 0.0
    assert rm.recall_at_k(rank, 3) == 1.0
    assert rm.reciprocal_rank(rank) == 0.5


def test_retrieval_metrics_doc_fallback():
    """A correct document but wrong chunk still matches via the doc fallback."""
    rank = rm.gold_rank(["x", "y"], ["d9", "dgold"], "missing", "dgold")
    assert rank == 2


def test_retrieval_metrics_doc_fallback_disabled():
    """With the fallback off, only an exact id match counts (visual/multihop)."""
    # Same-document sibling no longer rescues a missed exact id.
    assert (
        rm.gold_rank(
            ["x", "y"], ["d9", "dgold"], "missing", "dgold", allow_doc_fallback=False
        )
        is None
    )
    # An exact id match still counts regardless of the flag.
    assert (
        rm.gold_rank(
            ["x", "GOLD"], ["d9", "dgold"], "GOLD", "dgold", allow_doc_fallback=False
        )
        == 2
    )


def test_retrieval_aggregate_means():
    """Aggregation averages per-query metrics across the set."""
    metrics = rm.aggregate([1, None, 2], ks=[1, 3])
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert abs(metrics["recall@3"] - 2 / 3) < 1e-9


def _item(modality: str, ident: str, score: float = 0.0):
    """Builds a RetrievedItem for fusion tests."""
    from publaynet_mmrag.types import RetrievedItem

    kwargs = {"chunk_id": ident} if modality == "text" else {"region_id": ident}
    return RetrievedItem(
        doc_id="d", page_index=0, score=score, modality=modality, **kwargs
    )


def test_fuse_channels_single_channel_preserves_order():
    """One channel fuses to its own rank order (the baseline arm)."""
    from publaynet_mmrag.retrieve.multimodal_rag import fuse_channels

    text = [_item("text", f"c{i}") for i in range(4)]
    fused = fuse_channels([(1.0, text)], rrf_k=60)
    assert [it.chunk_id for it in fused] == ["c0", "c1", "c2", "c3"]


def test_fuse_channels_surfaces_image_into_topk():
    """With equal weights a top image hit interleaves into the top-k.

    This is the fix for the inert image channel: under the old raw-score merge a
    figure (small SigLIP cosine) sank below every text hit; RRF by rank lets it
    reach the top few.
    """
    from publaynet_mmrag.retrieve.multimodal_rag import fuse_channels

    text = [_item("text", f"c{i}") for i in range(5)]
    image = [_item("image", "GOLD"), _item("image", "r2")]
    fused = fuse_channels([(1.0, text), (1.0, image)], rrf_k=60)
    order = [it.chunk_id or it.region_id for it in fused]
    assert order.index("GOLD") < 5  # reaches the top-5 output cut


def test_fuse_channels_cross_channel_agreement_boosts():
    """An item returned by two channels outranks single-channel items."""
    from publaynet_mmrag.retrieve.multimodal_rag import fuse_channels

    text = [_item("text", "shared"), _item("text", "t1")]
    graph = [_item("text", "shared"), _item("text", "g1")]
    fused = fuse_channels([(1.0, text), (1.0, graph)], rrf_k=60)
    assert fused[0].chunk_id == "shared"


def _mini_graph():
    """A tiny graph: A & B co-occur in chunk C1; B alone in C2 (the bridge)."""
    builder = KnowledgeGraphBuilder(cooccurrence=True)
    long = " filler clause for length." * 12
    c1 = Chunk("D1:0:c0", "D1", 0, "alpha method and beta result." + long, ["r"])
    c2 = Chunk("D2:0:c0", "D2", 0, "beta result and gamma findings." + long, ["r"])
    builder.add_chunk(
        c1,
        [Entity("alpha method", "method", 0.9), Entity("beta result", "result", 0.9)],
    )
    builder.add_chunk(
        c2,
        [Entity("beta result", "result", 0.9), Entity("gamma signal", "task", 0.9)],
    )
    return builder.graph, {c1.chunk_id: c1, c2.chunk_id: c2}


def test_match_entities_respects_word_boundaries_and_length():
    """Whole-word matching only; sub-minimum names are ignored."""
    from publaynet_mmrag.kg import query as kg_query

    graph, _ = _mini_graph()
    # 'alpha method' is a whole-word hit; matching is case-insensitive.
    hit = kg_query._match_entities(graph, "What does Alpha Method achieve?")
    names = {graph.nodes[n].get("name").lower() for n in hit}
    assert "alpha method" in names
    # No spurious match when the surface form only appears inside a longer word.
    assert kg_query._match_entities(graph, "the alphamethodology was used") == []


def test_expand_ranks_bridge_chunk_first_by_proximity():
    """Graph expansion surfaces the bridge chunk and ranks by proximity."""
    from publaynet_mmrag.kg import query as kg_query

    graph, _ = _mini_graph()
    exp = kg_query.expand(graph, "What does alpha method achieve?", hops=1)
    assert "D2:0:c0" in exp.chunk_ids  # the bridge (mentions B, not A) is reached
    # The chunk mentioning the anchor directly outranks the one-hop bridge.
    assert exp.chunk_ids.index("D1:0:c0") < exp.chunk_ids.index("D2:0:c0")


def test_multihop_candidates_yield_dense_hard_bridge():
    """A candidate's bridge mentions B but not the anchor A (dense-hard)."""
    import random

    from publaynet_mmrag.eval.build_qa import _multihop_candidates

    graph, lookup = _mini_graph()
    cands = list(_multihop_candidates(graph, lookup, random.Random(0)))
    assert cands, "expected at least one candidate"
    cand = cands[0]
    assert cand["a"] == "alpha method"
    assert cand["b"] == "beta result"
    assert cand["chunk"].chunk_id == "D2:0:c0"
    # A is genuinely absent from the bridge graph node (no MENTIONS edge).
    a_node = "entity::method::alpha method"
    assert not graph.has_edge("D2:0:c0", a_node)


def test_normalise_row_upgrades_legacy_qa():
    """A legacy QA row (only gold_chunk_id) becomes a typed text row."""
    from publaynet_mmrag.eval.build_qa import normalise_row

    row = normalise_row({"question": "q", "gold_chunk_id": "c1", "gold_doc_id": "d"})
    assert row["qtype"] == "text"
    assert row["gold_id"] == "c1"
    assert row["gold_kind"] == "chunk"


def test_aggregate_scores_ignores_unscorable():
    """Per-sample judge scores mean correctly and skip None entries."""
    from publaynet_mmrag.eval.rag_metrics import aggregate_scores

    scored = [
        {"faithfulness": 1.0, "answer_relevancy": 0.5},
        {"faithfulness": None, "answer_relevancy": 1.0},
    ]
    out = aggregate_scores(scored)
    assert out["faithfulness"] == 1.0  # the None is skipped
    assert abs(out["answer_relevancy"] - 0.75) < 1e-9


def test_vision_generator_attaches_crops_and_parses_citations(tmp_path):
    """The vision generator attaches image-evidence crops and resolves citations."""
    from PIL import Image

    from publaynet_mmrag.config import GenerationConfig
    from publaynet_mmrag.reason.generate import VisionGenerator
    from publaynet_mmrag.types import RetrievedItem

    crop = tmp_path / "fig.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(crop)

    captured = {}

    class _StubVLM:
        def generate(self, system_prompt, user_text, images, temperature):
            captured["n_images"] = len(images)
            captured["user_text"] = user_text
            return "Reasoning: looked at the figure.\nAnswer: 4900 MFI. [S1]"

    items = [
        RetrievedItem(
            doc_id="D",
            page_index=2,
            score=1.0,
            modality="image",
            text="[table region]",
            region_id="D:2:r1",
            crop_path=str(crop),
            source="image",
        ),
        RetrievedItem(
            doc_id="D",
            page_index=2,
            score=0.9,
            modality="text",
            text="some context",
            chunk_id="D:2:c0",
            source="text_dense",
        ),
    ]
    gen = VisionGenerator(_StubVLM(), GenerationConfig())
    answer = gen.generate("What is the max MFI?", items, [])

    assert captured["n_images"] == 1  # the figure crop was attached
    assert "attached images" in captured["user_text"]  # vision note prepended
    # The (possibly wrong) caption is blanked so the VLM reads the image instead.
    assert "[table region]" not in captured["user_text"]
    assert "see attached image" in captured["user_text"]
    assert answer.text == "4900 MFI. [S1]"
    assert answer.citations == ["D:2:r1"]  # [S1] maps back to the image region


def test_kg_build_and_roundtrip():
    """The graph builds expected node/edge types and survives a save/load."""
    builder = KnowledgeGraphBuilder(cooccurrence=True)
    chunk = Chunk(
        chunk_id="DOC:0:c0", doc_id="DOC", page_index=0, text="t", region_ids=["r"]
    )
    entities = [
        Entity(text="BERT", label="model", score=0.9),
        Entity(text="SQuAD", label="dataset", score=0.8),
    ]
    triples = [Triple(subject="BERT", relation="evaluated_on", object="SQuAD")]
    builder.add_chunk(chunk, entities, triples)

    graph = builder.graph
    assert graph.has_node("DOC")
    assert graph.has_node("DOC:0:c0")
    assert any(d.get("ntype") == "entity" for _, d in graph.nodes(data=True))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kg.graphml")
        builder.save(path)
        reloaded = load_graph(path)
        assert reloaded.number_of_nodes() == graph.number_of_nodes()
