"""Data models shared by the local policy retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A policy document loaded from local storage."""

    document_id: str
    title: str
    source_path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A searchable portion of a policy document."""

    chunk_id: str
    document_id: str
    document_title: str
    source_path: str
    section: str
    text: str
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

