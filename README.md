# Autonomous arXiv Paper Digest & QA Agent

A small, interview-friendly AI internship assessment project that turns an arXiv paper into an executive briefing and then answers questions using grounded RAG.

## What it does

Input can be:

- an arXiv ID: `2401.12345`
- an arXiv URL: `https://arxiv.org/abs/2401.12345`
- a natural-language research topic: `retrieval augmented generation`

For a topic, the official arXiv API returns relevant candidates and the top arXiv relevance result is selected.

The selected paper is downloaded, parsed with PyMuPDF, chunked, embedded locally with HuggingFace, and stored in local ChromaDB. Groq-hosted LLM generates the briefing and answers QA questions from retrieved paper chunks.

## Architecture

```text
User input
   |
   v
resolve_input
   |
   +---- ID/URL -------------------+
   |                               |
   +---- topic -> arXiv API -> select
                                   |
                                   v
                            download_pdf
                                   |
                                   v
                              parse_pdf
                                   |
                                   v
                           index_chroma
                                   |
                                   v
                         generate_briefing
                                   |
                                  END

QA question
   |
   v
retrieve_qa
   |
   v
answer_qa
   |
  END

Both workflows use the same `PaperState`; QA receives the already indexed
paper identity from the ingestion state and performs fresh Chroma retrieval
for every question.
```

### Shared LangGraph state

`PaperState` carries:

- `user_input`
- `input_type`
- selected `paper`
- candidate papers
- PDF path
- parsed pages
- chunks
- briefing
- QA question
- retrieved chunks
- answer
- sources
- error

This makes the workflow explicit and stateful instead of hiding the whole application inside one LLM prompt.

## Repository structure

```text
arxiv_paper_digest_qa_agent/
├── app/
│   ├── __init__.py
│   ├── arxiv.py          # arXiv API, ID parsing, PDF download
│   ├── cli.py            # simple terminal interface
│   ├── config.py         # environment/configuration
│   ├── graph.py          # LangGraph nodes, edges and workflows
│   ├── llm.py            # Groq/Llama + structured outputs
│   ├── pdf_parser.py     # PyMuPDF extraction and chunking
│   ├── prompts.py        # grounding prompts
│   ├── retrieval.py      # local ChromaDB + HF embeddings
│   └── state.py          # shared graph state
├── tests/
│   ├── test_arxiv.py
│   ├── test_pdf_parser.py
│   └── test_prompts.py
├── data/
│   ├── chroma/           # generated locally, gitignored
│   └── papers/           # generated PDFs, gitignored
├── .env.example
├── .gitignore
├── .pytest.ini
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd arxiv_paper_digest_qa_agent

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_key
```

No paid vector database or paid embedding API is required. Embeddings run locally.

## Run

By arXiv ID:

```bash
python -m app.cli 1706.03762
```

By URL:

```bash
python -m app.cli https://arxiv.org/abs/1706.03762
```

By topic:

```bash
python -m app.cli "retrieval augmented generation"
```

Or run without an argument:

```bash
python -m app.cli
```

Then enter the paper ID, URL, or topic.

## Example output

```text
========================================================================
AUTONOMOUS arXiv PAPER DIGEST
========================================================================
Title: Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, ...
arXiv ID: 1706.03762
Published: 2017-06-12
Categories: cs.CL, cs.LG
Link: https://arxiv.org/abs/1706.03762

Why it matters:
The paper introduces the Transformer architecture...

Problem:
The authors address limitations of sequence transduction approaches
that rely heavily on recurrent or convolutional computation...

Method:
The model uses self-attention and feed-forward layers arranged in an
encoder-decoder architecture...

Key results/claims:
The paper reports strong translation results and improved parallelization...

Limitations:
The extracted paper evidence does not establish every limitation...
```

Exact generated text depends on the Llama model and retrieved evidence.

## QA examples

Example 1:

```text
Q> What architecture does the paper propose?
A> The paper proposes a Transformer encoder-decoder architecture based
on attention mechanisms.
Sources:
  - Page 1; Section: INTRODUCTION
  - Page 3; Section: MODEL ARCHITECTURE
```

Example 2:

```text
Q> What optimizer was used?
A> ...
Sources:
  - Page 5; Section: TRAINING
```

Example 3 — unsupported question:

```text
Q> What was the authors' favorite programming language?
A> I couldn't find this information in the paper.
```

The QA pipeline retrieves from ChromaDB before every answer. The answer prompt explicitly forbids outside knowledge and requires the fallback sentence when evidence is insufficient.

## Failure handling

The pipeline surfaces failures for:

- invalid arXiv IDs
- no arXiv search results
- arXiv API/network errors
- PDF download errors or non-PDF responses
- PDF parsing errors
- empty extracted text
- empty Chroma index
- embedding/indexing errors
- LLM/API/configuration errors

The CLI prints a clear error instead of silently producing a fabricated result.

## Design decisions and tradeoffs

### Why LangGraph?

The assessment specifically asks for an explicit stateful graph. Separate nodes make ingestion and QA easy to explain:

1. resolve input
2. search/select
3. download
4. parse
5. index
6. generate briefing
7. retrieve QA evidence
8. answer QA

This is more transparent than one large agent prompt.

### Why local HuggingFace embeddings?

They are free and avoid an embedding API dependency. `all-MiniLM-L6-v2` is small enough for a practical local assessment project.

### Why ChromaDB?

It provides a simple persistent local vector store without requiring cloud infrastructure.

### Why arXiv's API?

It is the authoritative source requested for topic search and provides paper metadata.

### Why top relevance result for topic selection?

It keeps the project deterministic and small. The arXiv API performs relevance sorting, so the application does not need a separate ranking agent.

### Why not multi-agent?

The assessment is better served by a clear stateful workflow. Multiple agents would increase complexity without improving the core demonstration.

## Limitations

- Topic selection uses the first result from arXiv's relevance ordering rather than a learned reranker.
- Retrieval is vector-only; BM25/hybrid retrieval is intentionally omitted to keep the project small.
- PDF parsing is text based; scanned/image-only PDFs may not work.
- Section detection is heuristic.
- The briefing is generated from retrieved chunks plus the abstract, so some paper details may not be represented in the briefing evidence.
- Local embedding inference uses CPU and may be slower on the first run because the model is downloaded and cached.
- LLM output quality depends on the configured Groq model and API availability.

## Future improvements

- Hybrid BM25 + vector retrieval.
- Cross-encoder reranking.
- Better section-aware PDF parsing.
- Query rewriting and multi-query retrieval.
- Retrieval/evidence evaluation metrics.
- Citation spans and exact source excerpts.
- Optional local LLM mode.
- Persistent graph checkpoints for multi-session workflows.
- A small web UI after the core assessment is stable.

## Tests

Run:

```bash
pytest -q
```

The tests intentionally focus on deterministic components such as arXiv ID parsing, chunk metadata, and grounding instructions. Network/API/LLM tests are avoided so the basic suite remains fast and free.

