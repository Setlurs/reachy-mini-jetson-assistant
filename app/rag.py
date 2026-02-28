"""RAG — embeddings, ChromaDB knowledge base, and retrieval in one module."""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional

import httpx
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


# ── Embedding functions ───────────────────────────────────────────

class LlamaCppEmbeddings(EmbeddingFunction):
    """OpenAI-compatible /v1/embeddings (llama-server --embeddings on GPU)."""

    def __init__(self, base_url: str = "http://localhost:8081", model: str = "bge-small-en-v1.5", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._dim: Optional[int] = None

    def _dimension(self) -> int:
        if self._dim:
            return self._dim
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.post(f"{self.base_url}/v1/embeddings", json={"model": self.model, "input": "dim"})
                if r.status_code == 200:
                    vec = (r.json().get("data") or [{}])[0].get("embedding", [])
                    if vec:
                        self._dim = len(vec)
                        return self._dim
        except Exception:
            pass
        return 384

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        out = []
        with httpx.Client(timeout=self.timeout) as c:
            for text in input:
                try:
                    r = c.post(f"{self.base_url}/v1/embeddings", json={"model": self.model, "input": text})
                    if r.status_code == 200:
                        items = r.json().get("data") or []
                        vec = items[0].get("embedding", []) if items else []
                        if self._dim is None and vec:
                            self._dim = len(vec)
                        out.append(vec)
                    else:
                        out.append([0.0] * self._dimension())
                except Exception:
                    out.append([0.0] * self._dimension())
        return out


class OllamaEmbeddings(EmbeddingFunction):
    """Embeddings via Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        with httpx.Client(timeout=self.timeout) as c:
            for text in input:
                try:
                    r = c.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text, "keep_alive": "1h"})
                    out.append(r.json().get("embedding", []) if r.status_code == 200 else [0.0] * 768)
                except Exception:
                    out.append([0.0] * 768)
        return out


class LocalMiniLMEmbeddings(EmbeddingFunction):
    """Small in-process MiniLM embeddings (CPU)."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"):
        self.model_name = model_name
        self._model = None

    def __call__(self, input: Documents) -> Embeddings:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(input, convert_to_numpy=True, device="cpu").tolist()


# ── Knowledge Base ────────────────────────────────────────────────

class KnowledgeBase:
    """ChromaDB document store with pluggable embeddings."""

    def __init__(
        self,
        persist_dir: str = "./data/chromadb",
        embedding_backend: str = "llamacpp",
        embedding_model: str = "bge-small-en-v1.5",
        embedding_base_url: str = "http://localhost:8081",
        chunk_size: int = 200,
        chunk_overlap: int = 20,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._persist_dir = persist_dir
        backend = (embedding_backend or "llamacpp").lower()
        if backend == "llamacpp":
            emb = LlamaCppEmbeddings(base_url=embedding_base_url, model=embedding_model)
            coll_name = "knowledge_llamacpp"
        elif backend == "ollama":
            emb = OllamaEmbeddings(model=embedding_model)
            coll_name = "knowledge"
        else:
            emb = LocalMiniLMEmbeddings(model_name=embedding_model)
            coll_name = "knowledge_local"

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=coll_name, embedding_function=emb, metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, text: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None) -> int:
        if not doc_id:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        chunks = self._chunk(text)
        self._collection.add(
            documents=chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc_id, "chunk": i, **(metadata or {})} for i in range(len(chunks))],
        )
        return len(chunks)

    def add_file(self, file_path: str) -> int:
        p = Path(file_path)
        return self.add_document(p.read_text(encoding="utf-8"), {"filename": p.name, "path": str(p)}, p.stem)

    def add_directory(self, dir_path: str, extensions: List[str] = None) -> int:
        exts = extensions or [".txt", ".md", ".py", ".json"]
        total = 0
        for f in Path(dir_path).rglob("*"):
            if f.is_file() and f.suffix in exts:
                try:
                    n = self.add_file(str(f))
                    total += n
                    print(f"  Added: {f.name} ({n} chunks)")
                except Exception as e:
                    print(f"  Error: {f.name}: {e}")
        return total

    def sync_directory(self, dir_path: str, extensions: List[str] = None) -> tuple[int, bool]:
        """Re-index only if knowledge base files changed. Returns (chunk_count, was_rebuilt)."""
        exts = extensions or [".txt", ".md", ".py", ".json"]
        d = Path(dir_path)
        hash_file = Path(self._persist_dir) / ".kb_hash"

        files = sorted(f for f in d.rglob("*") if f.is_file() and f.suffix in exts)
        h = hashlib.md5()
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        current_hash = h.hexdigest()

        if hash_file.exists() and hash_file.read_text().strip() == current_hash and self.count() > 0:
            return self.count(), False

        self.clear()
        total = self.add_directory(dir_path, extensions)
        hash_file.write_text(current_hash)
        return total, True

    def search(self, query: str, n_results: int = 3) -> List[dict]:
        r = self._collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
        if not r["documents"] or not r["documents"][0]:
            return []
        return [
            {"content": doc, "metadata": r["metadatas"][0][i], "distance": r["distances"][0][i]}
            for i, doc in enumerate(r["documents"][0])
        ]

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        name = self._collection.name
        emb = self._collection._embedding_function
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(name=name, embedding_function=emb, metadata={"hnsw:space": "cosine"})

    def _chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in text.replace("\n\n", ". ").replace(". ", ".\n").split("\n") if s.strip()]
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            j = i + 1
            while j < len(sentences) and len(chunk) + len(sentences[j]) + 1 <= self.chunk_size:
                chunk += " " + sentences[j]
                j += 1
            chunks.append(chunk)
            # step back by overlap: find how many trailing sentences fit in overlap window
            overlap_len = 0
            back = j - 1
            while back > i and overlap_len + len(sentences[back]) <= self.chunk_overlap:
                overlap_len += len(sentences[back])
                back -= 1
            i = max(back + 1, i + 1)
        return chunks or [text[:self.chunk_size]]


# ── Retriever ─────────────────────────────────────────────────────

class RAGRetriever:
    """Retrieves context from KnowledgeBase and formats it for LLM."""

    def __init__(self, kb: KnowledgeBase, n_results: int = 3, min_relevance: float = 0.5):
        self.kb = kb
        self.n_results = n_results
        self.min_relevance = min_relevance

    def augment_query(self, query: str) -> str:
        docs = self.kb.search(query, n_results=self.n_results)
        relevant = [d for d in docs if d.get("distance", 2) < (2 - self.min_relevance * 2)]
        if not relevant:
            return query
        ctx = "\n\n".join(d["content"] for d in relevant)
        return (
            "Context:\n" + ctx + "\n\n"
            "Answer the question using ONLY facts from the context above. "
            "Only answer what was asked, do not add extra information. "
            "If the context does not answer the question, say you do not know.\n\n"
            f"Question: {query}"
        )
