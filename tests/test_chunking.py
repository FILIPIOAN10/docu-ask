import pytest
from fpdf import FPDF

from docuask.chunking import EmptyDocument, UnsupportedDocument, chunk_sections, load

MD = b"# Policy\n\n## Sick leave\n\nTen days paid.\n\n## Carry-over\n\nFive days.\n"


def test_markdown_is_split_by_headings_and_keeps_the_section_path():
    sections = load("p.md", MD)
    assert [s.section for s in sections] == ["Policy > Sick leave", "Policy > Carry-over"]
    assert "Ten days paid." in sections[0].text


def test_the_embedded_text_starts_with_the_section_path():
    chunks = chunk_sections(load("p.md", MD), 800, 100)
    assert chunks[0].embed_text.startswith("Policy > Sick leave\n")
    assert chunks[0].content == "Ten days paid."  # what the user sees is the original text only


def test_long_sections_are_split_to_the_size_limit_with_overlap():
    body = " ".join(f"Sentence number {i} says something useful." for i in range(120)).encode()
    chunks = chunk_sections(load("long.txt", body), 400, 50)
    assert len(chunks) > 5
    assert all(len(c.content) <= 400 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_pdf_pages_become_sections_with_page_numbers():
    pdf = FPDF()
    for text in ("First page about leave.", "Second page about expenses."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text=text)
    chunks = chunk_sections(load("doc.pdf", bytes(pdf.output())), 800, 100)
    assert [(c.page, c.content) for c in chunks] == [(1, "First page about leave."), (2, "Second page about expenses.")]


def test_nul_characters_are_removed_because_postgres_rejects_them():
    chunks = chunk_sections(load("a.txt", b"hello\x00 world"), 800, 100)
    assert chunks[0].content == "hello world"


def test_unsupported_and_broken_files_are_refused():
    with pytest.raises(UnsupportedDocument):
        load("virus.exe", b"MZ")
    with pytest.raises(UnsupportedDocument):
        load("latin.txt", "Ü".encode("latin-1"))
    with pytest.raises(UnsupportedDocument):
        load("broken.pdf", b"not a pdf")


def test_a_document_without_text_is_refused():
    with pytest.raises(EmptyDocument):
        chunk_sections(load("empty.txt", b"   \n  "), 800, 100)
