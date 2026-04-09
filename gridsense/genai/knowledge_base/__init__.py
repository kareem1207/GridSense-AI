"""Knowledge base for RAG-powered transformer fault diagnosis."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(_PKG_DIR, "docs")
CHROMA_DIR = os.path.join(_PKG_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


class GridSenseKnowledgeBase:
    """ChromaDB knowledge base for RAG-powered transformer fault diagnosis.

    Stores five domain knowledge documents chunked into paragraphs.
    Uses sentence-transformers for embedding and ChromaDB for vector retrieval.
    All heavy imports (chromadb, sentence-transformers) are lazy.
    """

    def __init__(
        self,
        docs_dir: str = DOCS_DIR,
        persist_dir: str = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        """Initialise knowledge base paths (no IO performed here)."""
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazily initialise the ChromaDB PersistentClient and collection."""
        if self._client is None:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)
            self._collection = self._client.get_or_create_collection(
                name="gridsense_kb",
                embedding_function=embed_fn,
            )
        return self._client

    def build(self) -> None:
        """Load all .txt docs, chunk by paragraph, embed, and persist in ChromaDB."""
        self._get_client()
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            logger.warning("Docs directory not found: %s", self.docs_dir)
            return
        if self._collection.count() > 0:
            logger.info("Knowledge base already has %d chunks — skipping.", self._collection.count())
            return
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        txt_files = sorted(docs_path.glob("*.txt"))
        for txt_file in txt_files:
            content = txt_file.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 50]
            for j, para in enumerate(paragraphs):
                documents.append(para)
                ids.append(f"{txt_file.stem}_{j}")
                metadatas.append({"source": txt_file.name, "chunk": j})
        if documents:
            self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
            logger.info("Knowledge base built: %d chunks from %d files", len(documents), len(txt_files))

    def retrieve_similar_faults(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve the top-k most relevant knowledge chunks for a query."""
        self._get_client()
        if self._collection.count() == 0:
            self.build()
        if self._collection.count() == 0:
            return ["No knowledge base available. Using default diagnosis template."]
        n = min(top_k, self._collection.count())
        results = self._collection.query(query_texts=[query], n_results=n)
        return results.get("documents", [[]])[0]

    def is_built(self) -> bool:
        """Return True if the ChromaDB collection has been populated."""
        try:
            self._get_client()
            return self._collection.count() > 0
        except Exception:
            return False
