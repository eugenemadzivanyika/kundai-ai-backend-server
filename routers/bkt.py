from fastapi import APIRouter

from services.bkt_service import update_mastery

router = APIRouter(prefix="/bkt", tags=["bkt"])


@router.post("/update")
def update_bkt_mastery(payload: dict):
    # This receives the JSON from Node.js and returns the new mastery score
    return update_mastery(payload)
