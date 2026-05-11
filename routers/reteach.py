from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio

from services.reteach_service import generate_reteach_card

router = APIRouter(prefix="/reteach", tags=["Re-teach Cards"])


class TopicEntry(BaseModel):
    topic: str
    subtopic: str = ""
    avg_mastery: float
    student_count: int = 0


class GenerateReteachRequest(BaseModel):
    class_id: str
    subject: str
    topics: List[TopicEntry]


@router.post("/generate-cards")
async def generate_cards(payload: GenerateReteachRequest):
    if not payload.topics:
        return []

    try:
        tasks = [
            generate_reteach_card(
                topic=t.topic,
                subtopic=t.subtopic,
                subject=payload.subject,
                avg_mastery=t.avg_mastery,
            )
            for t in payload.topics
        ]
        cards = await asyncio.gather(*tasks)
        return list(cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
