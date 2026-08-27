"""Load public policy documents from local files."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


class _VisibleTextParser(HTMLParser):
    """Small dependency-free HTML-to-text parser for downloaded policy pages."""

    BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg"}
    INLINE_SEPARATOR_TAGS = {"a"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        elif not self.skip_depth and tag in self.INLINE_SEPARATOR_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        elif not self.skip_depth and tag in self.INLINE_SEPARATOR_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned).strip()

def _extract_policy_body(text: str) -> str:
    """Keep the public policy content and remove website navigation and footer."""

    lines = text.splitlines()
    start = None

    for index, line in enumerate(lines):
        if line.strip().lower() == "information classification - public":
            start = index

            for previous in range(index - 1, -1, -1):
                if lines[previous].strip():
                    start = previous
                    break
            break

    if start is None:
        return text

    end = len(lines)
    end_markers = (
        "Page contact:",
        "Powered by",
        "Let us know you agree to cookies",
    )

    for index in range(start + 1, len(lines)):
        if lines[index].strip().startswith(end_markers):
            end = index
            break

    return "\n".join(lines[start:end]).strip()

def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _load_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires PyMuPDF. Run: pip install -r requirements.txt"
        ) from exc

    pages: list[str] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _load_html(path: Path) -> str:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.text()


def load_document(path: str | Path) -> Document:
    """Load one supported policy document."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Policy file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported policy format {suffix!r}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        raw_text = _load_pdf(source)
    elif suffix in {".html", ".htm"}:
        raw_text = _load_html(source)
    else:
        raw_text = source.read_text(encoding="utf-8", errors="replace")

    text = _normalise_text(raw_text)

    if suffix in {".html", ".htm"}:
        text = _extract_policy_body(text)

    if not text:
        raise ValueError(f"No readable text was extracted from {source}")

    title = source.stem.replace("_", " ").replace("-", " ").strip().title()
    document_id = _slug(source.stem)
    return Document(
        document_id=document_id,
        title=title,
        source_path=str(source.resolve()),
        text=text,
        metadata={"suffix": suffix, "filename": source.name},
    )


def load_directory(data_dir: str | Path) -> list[Document]:
    """Load all supported files from a directory in deterministic order."""

    directory = Path(data_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Policy data directory not found: {directory}")

    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported policy files found in {directory}")
    return [load_document(path) for path in paths]

