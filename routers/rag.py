import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from services import pdf_parser_service, document_classifier_service, question_extractor_service, rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    subject_id: str
    query: str
    n_results: int = 5
    document_type: str | None = None


class AttributeItem(BaseModel):
    id: str
    name: str
    description: str = ""


class CoverageQueryRequest(BaseModel):
    subject_id: str
    attributes: list[AttributeItem]
    distance_threshold: float = 0.45


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/inject")
async def inject_document(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    document_id = str(uuid.uuid4())
    suffix = ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        text = await pdf_parser_service.extract_text_from_pdf(temp_path)
        doc_type = await document_classifier_service.classify_document(text[:6000])

        if doc_type == "question_paper":
            # Truncate to first 40 000 chars — enough to cover 150 questions without timing out
            questions = await question_extractor_service.extract_questions_from_text(text[:40000])
            chunks = [q.get("question_text", "") for q in questions if q.get("question_text")]
            metadata_base = {
                "source_file": file.filename,
                "document_id": document_id,
                "subject_id": subject_id,
                "document_type": "question_paper",
            }
            # Merge per-question hints into individual chunk metadata by doing inject manually
            if chunks:
                embeddings = await rag_service.embed_texts(chunks)
                collection = rag_service.get_or_create_collection(subject_id)
                ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        **metadata_base,
                        "chunk_index": i,
                        "marks_hint": questions[i].get("marks_hint"),
                        "topic_hint": questions[i].get("topic_hint"),
                    }
                    for i in range(len(chunks))
                ]
                collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
                chunks_stored = len(chunks)
            else:
                chunks_stored = 0
        else:
            chunks = rag_service.chunk_text(text)
            metadata_base = {
                "source_file": file.filename,
                "document_id": document_id,
                "subject_id": subject_id,
                "document_type": "learning_material",
            }
            chunks_stored = await rag_service.inject_document(
                subject_id, document_id, chunks, metadata_base
            )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return {
        "subject_id": subject_id,
        "document_id": document_id,
        "document_type": doc_type,
        "chunks_stored": chunks_stored,
        "filename": file.filename,
    }


@router.post("/query")
async def query_rag(body: QueryRequest):
    return await rag_service.query_rag(
        body.subject_id, body.query, body.n_results, body.document_type
    )


@router.post("/coverage-query")
async def coverage_query(body: CoverageQueryRequest):
    attrs = [{"id": a.id, "name": a.name, "description": a.description} for a in body.attributes]
    return await rag_service.compute_attribute_coverage(
        body.subject_id, attrs, body.distance_threshold
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, subject_id: str = Query(...)):
    try:
        collection = rag_service.get_or_create_collection(subject_id)
        existing = collection.get(where={"document_id": document_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return {"deleted": len(ids_to_delete), "document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{subject_id}")
async def rag_status(subject_id: str):
    try:
        collection = rag_service.get_or_create_collection(subject_id)
        count = collection.count()
        return {"subject_id": subject_id, "chunk_count": count, "has_documents": count > 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
