"""Chunk policies using either document structure or fixed word windows."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Chunk, Document


NUMBERED_SECTION = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+){0,3})[.)]?\s+(?P<title>\S.*)$"
)
MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+(?P<title>\S.*)$")
PAGE_MARKER = re.compile(r"^\[Page\s+\d+\]$", re.IGNORECASE)


def _word_windows(words: list[str], max_words: int, overlap_words: int) -> Iterable[list[str]]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    step = max_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            yield window
        if start + max_words >= len(words):
            break


def _make_chunk(
    document: Document,
    index: int,
    text: str,
    section: str,
    strategy: str,
) -> Chunk:
    words = text.split()
    return Chunk(
        chunk_id=f"{document.document_id}-{strategy}-{index:04d}",
        document_id=document.document_id,
        document_title=document.title,
        source_path=document.source_path,
        section=section,
        text=text.strip(),
        word_count=len(words),
        metadata={"strategy": strategy},
    )


def chunk_fixed(
    document: Document,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Split a document into overlapping fixed-size word windows."""

    words = document.text.split()
    return [
        _make_chunk(document, index, " ".join(window), "Fixed window", "fixed")
        for index, window in enumerate(
            _word_windows(words, max_words=max_words, overlap_words=overlap_words),
            start=1,
        )
    ]


def _is_section_start(line: str) -> tuple[bool, str]:
    numbered = NUMBERED_SECTION.match(line)
    if numbered:
        return True, f"{numbered.group('number')} {numbered.group('title')}"

    markdown = MARKDOWN_HEADING.match(line)
    if markdown:
        return True, markdown.group("title")

    stripped = line.strip()
    if 2 <= len(stripped.split()) <= 10 and stripped.isupper():
        return True, stripped.title()

    return False, ""


def _policy_blocks(text: str) -> list[tuple[str, str]]:
    """Return `(section_label, block_text)` pairs from policy-like text."""

    blocks: list[tuple[str, str]] = []
    current_label = "Document introduction"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((current_label, block_text))
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue
        if PAGE_MARKER.match(stripped):
            continue

        is_start, label = _is_section_start(stripped)
        if is_start:
            flush()
            current_label = label
        current_lines.append(stripped)

    flush()
    return blocks


def chunk_policy_aware(
    document: Document,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Chunk a policy while preserving numbered clauses and headings."""

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    chunks: list[Chunk] = []
    index = 1
    for label, block_text in _policy_blocks(document.text):
        block_words = block_text.split()
        for window in _word_windows(block_words, max_words, overlap_words):
            chunks.append(
                _make_chunk(document, index, " ".join(window), label, "policy")
            )
            index += 1
    return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Apply one named chunking strategy to multiple documents."""

    if strategy not in {"policy", "fixed"}:
        raise ValueError("strategy must be either 'policy' or 'fixed'")

    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "policy":
            chunks.extend(chunk_policy_aware(document, max_words, overlap_words))
        else:
            chunks.extend(chunk_fixed(document, max_words, overlap_words))
    return chunks
