from fastapi import APIRouter, HTTPException

from services.developmentPlan_service import generate_surgical_missions

router = APIRouter(prefix="/developmentplan", tags=["Remediation Agent"])

@router.post("/generate-missions")
async def generate_missions(payload: dict):
    # Payload from Node.js: { "attribute_id": "MATH-F1-RN-02", "initial_mastery": 0.1223 }
    try:
        result = await generate_surgical_missions(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
