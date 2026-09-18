from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from .prompts import BRIEFING_SYSTEM, BRIEFING_USER, QA_SYSTEM, QA_USER

FALLBACK = "I couldn't find this information in the paper."

class Briefing(BaseModel):
    why_it_matters: str
    problem: str
    method: str
    key_results_or_claims: str
    limitations: str
    suggested_questions: list[str] = Field(min_length=1, max_length=5)

class QAResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)

def get_llm(api_key: str, model: str):
    return ChatGroq(api_key=api_key, model=model, temperature=0)

def generate_briefing(llm, paper: dict, chunks: list[dict]) -> dict:
    evidence = "\n\n".join(
        f"[Page {c.get('page')}; Section: {c.get('section')}] {c['text']}"
        for c in chunks[:12]
    )
    result = llm.with_structured_output(Briefing).invoke([
        ("system", BRIEFING_SYSTEM),
        ("human", BRIEFING_USER.format(
            metadata=paper,
            abstract=paper.get("abstract", ""),
            evidence=evidence,
        )),
    ])
    return result.model_dump()

def answer_question(llm, question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": FALLBACK, "sources": []}

    evidence = "\n\n".join(
        f"[Page {c.get('page')}; Section: {c.get('section')}] {c['text']}"
        for c in chunks
    )
    result = llm.with_structured_output(QAResponse).invoke([
        ("system", QA_SYSTEM),
        ("human", QA_USER.format(question=question, evidence=evidence)),
    ]).model_dump()

    answer = result.get("answer", "").strip()
    if not answer or FALLBACK.lower() in answer.lower():
        return {"answer": FALLBACK, "sources": []}

    valid_pages = {str(c.get("page")) for c in chunks}
    valid_sources = [
        str(source).strip()
        for source in result.get("sources", [])
        if any(page in str(source) for page in valid_pages)
    ]
    # If the model cannot provide a source tied to retrieved paper pages,
    # do not present an unsupported answer as grounded.
    if not valid_sources:
        return {"answer": FALLBACK, "sources": []}

    return {"answer": answer, "sources": valid_sources}
