from app.pdf_parser import chunk_pages, infer_section

def test_infer_section():
    assert infer_section("INTRODUCTION\nThis is the paper.") == "INTRODUCTION"

def test_chunk_pages_has_metadata():
    chunks = chunk_pages([{"page": 3, "text": "word " * 500}], chunk_size=100, chunk_overlap=20)
    assert chunks
    assert chunks[0]["page"] == 3
    assert "section" in chunks[0]
