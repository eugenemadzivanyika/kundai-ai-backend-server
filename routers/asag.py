from fastapi import APIRouter, HTTPException
from services.asag_service import grade_student_work

# We use the /asag prefix to match your Node.js axios calls
router = APIRouter(
    prefix="/asag",
    tags=["Automated Student Answer Grading"]
)

@router.post("/grade")
async def perform_grading(payload: dict):
    """
    Endpoint to trigger Chain-of-Thought AI grading for subjective work.
    Expects: content (student text), rubric (question + keywords), studentContext.
    """
    try:
        # This calls the service that uses the ZIMSEC Moderator persona
        result = await grade_student_work(payload)
        return result
    except Exception as e:
        print(f"[GRADING ROUTE ERROR]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI Grading failed: {str(e)}"
        )