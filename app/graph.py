from __future__ import annotations

from pathlib import Path
from langgraph.graph import StateGraph, END
from .arxiv import normalize_id, paper_by_id, search_topic, download_pdf, paper_to_dict, ArxivPaper
from .config import get_settings
from .pdf_parser import extract_pages, chunk_pages
from .retrieval import ChromaRetriever
from .llm import get_llm, generate_briefing, answer_question
from .state import PaperState

def resolve_input(state: PaperState) -> dict:
    value = state["user_input"].strip()
    arxiv_id = normalize_id(value)
    if arxiv_id:
        paper = paper_by_id(arxiv_id, get_settings().request_timeout)
        return {"input_type": "id", "paper": paper_to_dict(paper)}
    return {"input_type": "topic"}

def search_paper(state: PaperState) -> dict:
    if state.get("input_type") != "topic":
        return {}
    settings = get_settings()
    papers = search_topic(
        state["user_input"],
        max_results=settings.max_topic_results,
        timeout=settings.request_timeout,
    )
    if not papers:
        raise ValueError("No arXiv papers found for this topic.")
    # arXiv supplies relevance ordering; selecting the first result keeps
    # topic selection deterministic and avoids an unnecessary ranking agent.
    selected = papers[0]
    return {
        "candidate_papers": [paper_to_dict(p) for p in papers],
        "paper": paper_to_dict(selected),
    }

def download_node(state: PaperState) -> dict:
    settings = get_settings()
    paper_data = state["paper"]
    paper = ArxivPaper(**paper_data)
    path = Path("data/papers") / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    download_pdf(paper, path, timeout=settings.request_timeout)
    return {"pdf_path": str(path)}

def parse_node(state: PaperState) -> dict:
    settings = get_settings()
    pages = extract_pages(state["pdf_path"])
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    return {"pages": pages, "chunks": chunks}

def index_node(state: PaperState) -> dict:
    settings = get_settings()
    retriever = ChromaRetriever(settings.chroma_dir, settings.embedding_model)
    count = retriever.index(state["paper"]["arxiv_id"], state["chunks"])
    if count == 0:
        raise ValueError("No chunks were indexed.")
    return {}

def briefing_node(state: PaperState) -> dict:
    settings = get_settings()
    llm = get_llm(settings.groq_api_key, settings.groq_model)
    retriever = ChromaRetriever(settings.chroma_dir, settings.embedding_model)
    evidence = retriever.retrieve(
        state["paper"]["arxiv_id"],
        state["paper"]["abstract"],
        top_k=settings.top_k,
    )
    if not evidence:
        raise ValueError("No evidence could be retrieved for briefing generation.")
    return {"briefing": generate_briefing(llm, state["paper"], evidence)}

def retrieve_qa_node(state: PaperState) -> dict:
    settings = get_settings()
    retriever = ChromaRetriever(settings.chroma_dir, settings.embedding_model)
    chunks = retriever.retrieve(
        state["paper"]["arxiv_id"],
        state["question"],
        top_k=settings.top_k,
    )
    return {"retrieved_chunks": chunks}

def qa_node(state: PaperState) -> dict:
    settings = get_settings()
    llm = get_llm(settings.groq_api_key, settings.groq_model)
    result = answer_question(llm, state["question"], state.get("retrieved_chunks", []))
    return {"answer": result["answer"], "sources": result["sources"]}

def build_ingestion_graph():
    graph = StateGraph(PaperState)
    graph.add_node("resolve_input", resolve_input)
    graph.add_node("search_paper", search_paper)
    graph.add_node("download_pdf", download_node)
    graph.add_node("parse_pdf", parse_node)
    graph.add_node("index_chroma", index_node)
    graph.add_node("generate_briefing", briefing_node)

    graph.set_entry_point("resolve_input")
    graph.add_edge("resolve_input", "search_paper")
    graph.add_edge("search_paper", "download_pdf")
    graph.add_edge("download_pdf", "parse_pdf")
    graph.add_edge("parse_pdf", "index_chroma")
    graph.add_edge("index_chroma", "generate_briefing")
    graph.add_edge("generate_briefing", END)
    return graph.compile()

def build_qa_graph():
    graph = StateGraph(PaperState)
    graph.add_node("retrieve_qa", retrieve_qa_node)
    graph.add_node("answer_qa", qa_node)
    graph.set_entry_point("retrieve_qa")
    graph.add_edge("retrieve_qa", "answer_qa")
    graph.add_edge("answer_qa", END)
    return graph.compile()

def run_ingestion(user_input: str) -> PaperState:
    try:
        return build_ingestion_graph().invoke({"user_input": user_input})
    except Exception as exc:
        return {"user_input": user_input, "error": f"{type(exc).__name__}: {exc}"}

def run_qa(state: PaperState, question: str) -> PaperState:
    qa_state: PaperState = {
        "user_input": state["user_input"],
        "paper": state["paper"],
        "question": question,
    }
    try:
        result = build_qa_graph().invoke(qa_state)
        return {**state, **result}
    except Exception as exc:
        return {
            **state,
            "question": question,
            "error": f"{type(exc).__name__}: {exc}",
        }
