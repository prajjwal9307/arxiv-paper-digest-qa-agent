from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/{id}"
ARXIV_PDF = "https://arxiv.org/pdf/{id}.pdf"

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    categories: list[str]
    abs_url: str
    pdf_url: str

def normalize_id(value: str) -> str | None:
    value = value.strip()
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"^(?:arXiv:)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)$",
        r"arxiv\.org/abs/([a-z-]+/\d{7}(?:v\d+)?)",
        r"^(?:arXiv:)?([a-z-]+/\d{7}(?:v\d+)?)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, value, re.I)
        if m:
            return m.group(1)
    return None

def _parse_entry(entry: ET.Element) -> ArxivPaper:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    raw_id = entry.findtext("a:id", namespaces=ns) or ""
    arxiv_id = raw_id.rstrip("/").split("/")[-1]
    title = " ".join((entry.findtext("a:title", namespaces=ns) or "").split())
    abstract = " ".join((entry.findtext("a:summary", namespaces=ns) or "").split())
    published = entry.findtext("a:published", namespaces=ns) or ""
    authors = [
        (a.findtext("a:name", namespaces=ns) or "").strip()
        for a in entry.findall("a:author", ns)
    ]
    categories = [
        c.attrib.get("term", "")
        for c in entry.findall("a:category", ns)
        if c.attrib.get("term")
    ]
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        published=published[:10],
        abstract=abstract,
        categories=categories,
        abs_url=ARXIV_ABS.format(id=arxiv_id),
        pdf_url=ARXIV_PDF.format(id=arxiv_id),
    )

def paper_by_id(arxiv_id: str, timeout: int = 30) -> ArxivPaper:
    params = {"id_list": arxiv_id, "max_results": 1}
    response = requests.get(ARXIV_API, params=params, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    if not entries:
        raise ValueError(f"No arXiv paper found for ID: {arxiv_id}")
    return _parse_entry(entries[0])

def search_topic(topic: str, max_results: int = 5, timeout: int = 30) -> list[ArxivPaper]:
    params = {
        "search_query": f"all:{quote_plus(topic)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API, params=params, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [_parse_entry(e) for e in root.findall("a:entry", ns)]

def download_pdf(paper: ArxivPaper, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(paper.pdf_url, timeout=timeout, headers={"User-Agent": "ArxivDigestQA/1.0"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("Downloaded content is not a valid PDF.")
    destination.write_bytes(response.content)
    return destination

def paper_to_dict(paper: ArxivPaper) -> dict:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "published": paper.published,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
    }
