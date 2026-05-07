from __future__ import annotations

import os
import math
import asyncio
import httpx
import chromadb
from fastapi import HTTPException

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"
EMBED_BATCH_SIZE = 8
MAX_CHUNK_CHARS = 2000  # ~500 tokens; keeps total batch tokens within Mistral's 16 384 limit
CHROMADB_PATH = "./data/chromadb"

_chroma_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMADB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return _chroma_client


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 60) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

async def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    num_batches = math.ceil(len(texts) / batch_size)

    async with httpx.AsyncClient() as client:
        for i in range(num_batches):
            batch = [t[:MAX_CHUNK_CHARS] for t in texts[i * batch_size:(i + 1) * batch_size]]
            payload = {"model": EMBED_MODEL, "input": batch}
            # Retry up to 4 times with exponential backoff on 429
            for attempt in range(4):
                try:
                    response = await client.post(
                        MISTRAL_EMBED_URL, json=payload, headers=headers, timeout=60.0
                    )
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        print(f"Mistral embed 429 — retrying in {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except httpx.HTTPStatusError as e:
                    print(f"Mistral embed error {e.response.status_code}: {e.response.text}")
                    raise HTTPException(status_code=502, detail="Embedding API request failed")
                except Exception as e:
                    print(f"Embed error: {e}")
                    raise HTTPException(status_code=502, detail="Embedding request failed")
            else:
                raise HTTPException(status_code=502, detail="Embedding API rate limit — all retries exhausted")

    return all_embeddings


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_or_create_collection(subject_id: str) -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=f"subject_{subject_id}",
        metadata={"hnsw:space": "l2"},
    )


# ---------------------------------------------------------------------------
# Inject
# ---------------------------------------------------------------------------

async def inject_document(
    subject_id: str,
    document_id: str,
    chunks: list[str],
    metadata_base: dict,
) -> int:
    if not chunks:
        return 0

    embeddings = await embed_texts(chunks)
    collection = get_or_create_collection(subject_id)

    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata_base, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

async def query_rag(
    subject_id: str,
    query: str,
    n_results: int = 5,
    document_type: str | None = None,
) -> dict:
    try:
        collection = get_or_create_collection(subject_id)
        if collection.count() == 0:
            return {"chunks": [], "sources": [], "distances": [], "has_documents": False}

        query_embedding = await embed_texts([query])

        where = {"document_type": document_type} if document_type else None
        kwargs: dict = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        chunks = results["documents"][0] if results["documents"] else []
        sources = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return {
            "chunks": chunks,
            "sources": sources,
            "distances": distances,
            "has_documents": bool(chunks),
        }
    except Exception as e:
        print(f"query_rag error: {e}")
        return {"chunks": [], "sources": [], "distances": [], "has_documents": False}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

async def compute_attribute_coverage(
    subject_id: str,
    attributes: list[dict],
    distance_threshold: float = 0.35,
) -> list[dict]:
    if not attributes:
        return []

    try:
        collection = get_or_create_collection(subject_id)
        if collection.count() == 0:
            return [
                {"attribute_id": a["id"], "covered": False, "source_files": [], "max_relevance": 0.0}
                for a in attributes
            ]

        # Embed all attribute queries in a single API call — one request regardless of count
        query_strings = [
            f"{a['name']} {a.get('description', '')}".strip() for a in attributes
        ]
        query_embeddings = await embed_texts(query_strings, batch_size=32)

        n = min(3, collection.count())
        where = {"document_type": "learning_material"}

        results: list[dict] = []
        for attr, q_embedding in zip(attributes, query_embeddings):
            try:
                res = collection.query(
                    query_embeddings=[q_embedding],
                    n_results=n,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                chunks   = res["documents"][0]  if res["documents"]  else []
                sources  = res["metadatas"][0]  if res["metadatas"]  else []
                distances = res["distances"][0] if res["distances"]  else []

                hitting = [
                    (c, m, d) for c, m, d in zip(chunks, sources, distances)
                    if d < distance_threshold
                ]
                covered = bool(hitting)
                source_files = list({m.get("source_file", "") for _, m, _ in hitting if m.get("source_file")})
                max_relevance = round(1.0 - min(d for _, _, d in hitting), 4) if hitting else 0.0

                results.append({
                    "attribute_id": attr["id"],
                    "covered": covered,
                    "source_files": source_files,
                    "max_relevance": max_relevance,
                })
            except Exception as e:
                print(f"Coverage check error for attr {attr.get('id')}: {e}")
                results.append({
                    "attribute_id": attr.get("id", ""),
                    "covered": False,
                    "source_files": [],
                    "max_relevance": 0.0,
                })

        return results

    except Exception as e:
        print(f"compute_attribute_coverage error: {e}")
        return [
            {"attribute_id": a.get("id", ""), "covered": False, "source_files": [], "max_relevance": 0.0}
            for a in attributes
        ]


# ---------------------------------------------------------------------------
# Format context for LLM
# ---------------------------------------------------------------------------

def format_rag_context(rag_result: dict) -> str:
    if not rag_result.get("has_documents"):
        return ""

    lines = ["═══ CURRICULUM DOCUMENTS (grounding context) ═══"]
    for i, (chunk, source) in enumerate(
        zip(rag_result["chunks"], rag_result["sources"]), start=1
    ):
        filename = source.get("source_file", "unknown")
        lines.append(f"\n[Chunk {i} — source: {filename}]")
        lines.append(chunk)
    lines.append("\n═══ END OF GROUNDING CONTEXT ═══")
    return "\n".join(lines)
