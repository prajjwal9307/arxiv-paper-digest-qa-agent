from typing import Any, TypedDict

class PaperState(TypedDict, total=False):
    user_input: str
    input_type: str
    paper: dict[str, Any]
    candidate_papers: list[dict[str, Any]]
    pdf_path: str
    pages: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    briefing: dict[str, Any]
    question: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    error: str
