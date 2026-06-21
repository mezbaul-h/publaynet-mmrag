"""Dataset ingestion: a swappable source interface and two backends."""

from publaynet_mmrag.ingest.base import DocumentSource, PageSample, build_source

__all__ = ["DocumentSource", "PageSample", "build_source"]
