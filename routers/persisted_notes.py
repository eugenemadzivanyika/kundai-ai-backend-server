from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from urllib.parse import quote
import httpx
import os
import logging

from services.notes_service import generate_notes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/persisted-notes", tags=["Persisted Notes"])

NODE_API_URL = os.getenv("NODE_API_URL", "http://localhost:5000/api")


class GenerateNotesRequest(BaseModel):
    subject_id: str
    topic_name: str
    subtopic_name: str
    level: str = "O Level"
    school_id: Optional[str] = None
    syllabus_context: Optional[str] = None
    auth_token: Optional[str] = None


@router.post("/generate")
async def generate_and_persist_notes(body: GenerateNotesRequest):
    try:
        result = await generate_notes(
            subject_id=body.subject_id,
            topic=body.subtopic_name,
            attribute_name=body.subtopic_name,
            level=body.level,
        )
        content = result.get("notes", "")

        # Persist back to Node backend — non-fatal if it fails
        persisted = False
        try:
            topic_seg = quote(body.topic_name, safe="")
            sub_seg = quote(body.subtopic_name, safe="")
            url = (
                f"{NODE_API_URL}/subject-notes/{body.subject_id}"
                f"/topics/{topic_seg}/subtopics/{sub_seg}"
            )
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if body.auth_token:
                headers["Authorization"] = f"Bearer {body.auth_token}"
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json={"content": content}, headers=headers, timeout=15)
                persisted = resp.status_code == 200
                if not persisted:
                    logger.warning("Persist failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as persist_err:
            logger.warning("Persist callback error: %s", persist_err)

        return {
            "content": content,
            "persisted": persisted,
            "sources": result.get("sources", []),
            "grounded_by_rag": result.get("grounded_by_rag", False),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
