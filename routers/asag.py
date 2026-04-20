from fastapi import APIRouter, HTTPException
# Import both functions
from services.asag_service import grade_student_work, build_cognitive_profile

router = APIRouter(
    prefix="/asag",
    tags=["Automated Student Answer Grading"]
)

@router.post("/grade")
async def perform_grading_and_profiling(payload: dict):
    """
    Sequentially triggers Grading (Scoring) then Profiling (Understanding).
    """
    try:
        # STEP 1: Get the score and basic feedback
        grading_result = await grade_student_work(payload)
        
        # STEP 2: Pass the original payload AND the grading result to the profiler
        cognitive_profile = await build_cognitive_profile(payload, grading_result)
        
        # STEP 3: Return a unified object for the Node.js backend
        return {
            "grading": grading_result,
            "profile": cognitive_profile
        }
    except Exception as e:
        print(f"[PIPELINE ERROR]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI Pipeline failed: {str(e)}"
        )