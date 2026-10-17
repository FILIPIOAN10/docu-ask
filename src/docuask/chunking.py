"""From raw bytes to chunks: load -> split by structure -> split by size.

Splitting by structure first (Markdown headings, PDF pages) means a chunk rarely mixes two topics, and every chunk
knows where it came from (section path, page), which is what the citation shows the user.
"""

import io
from dataclasses import dataclass
from pathlib import PurePath

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader


class UnsupportedDocument(ValueError):
    """The file type is not one we can read (message is safe to show to the client)."""


class EmptyDocument(ValueError):
    """The file was read but contains no text (a scanned PDF without OCR, for example)."""


@dataclass(frozen=True)
class Section:
    text: str
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class ChunkDraft:
    ord: int
    content: str
    section: str | None
    page: int | None

    @property
    def embed_text(self) -> str:
        """What gets embedded: the section path in front of the text. 'Sick leave > 10 days' alone says little;
        with the heading in front, the vector knows what the 10 days are about."""
        return f"{self.section}\n{self.content}" if self.section else self.content


def _clean(text: str) -> str:
    # Postgres refuses the NUL character in text columns; pypdf emits it for some fonts.
    return text.replace("\x00", "").strip()


def load(name: str, data: bytes) -> list[Section]:
    suffix = PurePath(name).suffix.lower()
    if suffix in (".md", ".markdown"):
        return _load_markdown(_decode(data))
    if suffix == ".txt":
        return [Section(_clean(_decode(data)))]
    if suffix == ".pdf":
        return _load_pdf(data)
    raise UnsupportedDocument(f"Unsupported file type '{suffix or name}'. Use .md, .txt or .pdf")


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise UnsupportedDocument("The file is not valid UTF-8 text") from e


def _load_markdown(text: str) -> list[Section]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True
    )
    sections = []
    for doc in splitter.split_text(_clean(text)):
        path = " > ".join(doc.metadata[h] for h in ("h1", "h2", "h3") if h in doc.metadata)
        if doc.page_content.strip():
            sections.append(Section(doc.page_content, path or None))
    return sections


def _load_pdf(data: bytes) -> list[Section]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [Section(_clean(p.extract_text() or ""), page=i) for i, p in enumerate(reader.pages, start=1)]
    except Exception as e:  # pypdf raises many different types on a broken file
        raise UnsupportedDocument("The PDF could not be read") from e


def chunk_sections(sections: list[Section], size: int, overlap: int) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    drafts: list[ChunkDraft] = []
    for sec in sections:
        for piece in splitter.split_text(sec.text):
            if piece.strip():
                drafts.append(ChunkDraft(len(drafts), piece.strip(), sec.section, sec.page))
    if not drafts:
        raise EmptyDocument("The document has no readable text")
    return drafts
