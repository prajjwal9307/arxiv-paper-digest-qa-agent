from __future__ import annotations

from pathlib import Path
import fitz

def extract_pages(pdf_path: str | Path) -> list[dict]:
    pages: list[dict] = []
    doc = fitz.open(pdf_path)
    try:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_no, "text": text})
    finally:
        doc.close()
    if not pages:
        raise ValueError("PDF was parsed successfully but contained no extractable text.")
    return pages

def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[dict]:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    chunks: list[dict] = []
    for page in pages:
        text = " ".join(page["text"].split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "section": infer_section(page["text"]),
                })
            if end >= len(text):
                break
            start = end - chunk_overlap
    if not chunks:
        raise ValueError("No chunks could be created from extracted text.")
    return chunks

def infer_section(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:15]:
        normalized = " ".join(line.split())
        if 2 <= len(normalized) <= 90 and (
            normalized.isupper()
            or normalized.lower().startswith((
                "abstract", "introduction", "related work", "method",
                "methodology", "experiments", "results", "discussion",
                "conclusion", "limitations", "references"
            ))
        ):
            return normalized
    return "Unknown"
