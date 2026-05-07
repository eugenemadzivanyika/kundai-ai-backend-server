from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.notes_service import generate_notes, answer_topic_question

router = APIRouter(prefix="/notes", tags=["Notes"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class NotesRequest(BaseModel):
    subject_id: str
    topic: str
    attribute_name: str
    level: str
    student_profile: Optional[dict] = None


class TopicChatRequest(BaseModel):
    subject_id: str
    topic: str
    notes_context: str          # The Markdown the student is currently reading
    question: str
    history: list[dict] = []    # [{role: "user"|"assistant", content: str}]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate_notes_endpoint(body: NotesRequest):
    try:
        return await generate_notes(
            subject_id=body.subject_id,
            topic=body.topic,
            attribute_name=body.attribute_name,
            level=body.level,
            student_profile=body.student_profile,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def topic_chat_endpoint(body: TopicChatRequest):
    """
    Notes-aware topic chat (Phase 1).

    Accepts the Markdown notes the student is currently reading and answers
    their question using those notes as the primary context. Prior turns in
    `history` are forwarded to the LLM so the conversation is coherent.

    Returns: { answer: str, grounded_by_rag: bool }
    """
    try:
        return await answer_topic_question(
            subject_id=body.subject_id,
            topic=body.topic,
            notes_context=body.notes_context,
            question=body.question,
            history=body.history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))