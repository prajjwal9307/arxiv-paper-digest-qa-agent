from __future__ import annotations

from pathlib import Path
from typing import Any
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaRetriever:
    def __init__(self, persist_dir: str, embedding_model: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _collection(self, paper_id: str):
        safe = "".join(c if c.isalnum() else "_" for c in paper_id)
        return self.client.get_or_create_collection(
            name=f"paper_{safe}"[:63],
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, paper_id: str, chunks: list[dict[str, Any]]) -> int:
        collection = self._collection(paper_id)
        ids = [f"{paper_id}-{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "paper_id": paper_id,
                "page": c.get("page", 0),
                "section": c.get("section", "Unknown"),
                "chunk_index": i,
            }
            for i, c in enumerate(chunks)
        ]
        vectors = self.embeddings.embed_documents(documents)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )
        return len(ids)

    def retrieve(self, paper_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        collection = self._collection(paper_id)
        if collection.count() == 0:
            return []
        vector = self.embeddings.embed_query(query)
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "text": doc,
                "page": meta.get("page"),
                "section": meta.get("section"),
                "distance": distance,
            }
            for doc, meta, distance in zip(docs, metas, distances)
        ]
