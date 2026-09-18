from __future__ import annotations

import argparse
from .config import get_settings
from .graph import run_ingestion, run_qa

def print_briefing(state: dict):
    paper = state["paper"]
    b = state["briefing"]
    print("\n" + "=" * 72)
    print("AUTONOMOUS arXiv PAPER DIGEST")
    print("=" * 72)
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"arXiv ID: {paper['arxiv_id']}")
    print(f"Published: {paper['published']}")
    print(f"Categories: {', '.join(paper['categories'])}")
    print(f"Link: {paper['abs_url']}")
    print(f"\nWhy it matters:\n{b['why_it_matters']}")
    print(f"\nProblem:\n{b['problem']}")
    print(f"\nMethod:\n{b['method']}")
    print(f"\nKey results/claims:\n{b['key_results_or_claims']}")
    print(f"\nLimitations:\n{b['limitations']}")
    print("\nSuggested questions:")
    for i, q in enumerate(b["suggested_questions"], 1):
        print(f"  {i}. {q}")

def main():
    parser = argparse.ArgumentParser(description="Autonomous arXiv Paper Digest & QA Agent")
    parser.add_argument("input", nargs="*", help="arXiv ID, arXiv URL, or research topic")
    args = parser.parse_args()

    user_input = " ".join(args.input).strip()
    if not user_input:
        user_input = input("Enter arXiv ID, arXiv URL, or research topic: ").strip()
    if not user_input:
        print("Input cannot be empty.")
        return

    try:
        get_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}")
        print("Create .env from .env.example and set GROQ_API_KEY.")
        return

    print("\nRunning stateful LangGraph pipeline...")
    state = run_ingestion(user_input)
    if state.get("error"):
        print(f"\nPipeline failed: {state['error']}")
        return

    print_briefing(state)
    print("\n" + "-" * 72)
    print("QA MODE — ask questions about the selected paper. Type 'exit' to stop.")

    while True:
        question = input("\nQ> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = run_qa(state, question)
        if result.get("error"):
            print(f"Error: {result['error']}")
            continue
        print(f"\nA> {result['answer']}")
        if result.get("sources"):
            print("Sources:")
            for source in result["sources"]:
                print(f"  - {source}")

if __name__ == "__main__":
    main()
