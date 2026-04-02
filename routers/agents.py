from fastapi import APIRouter, HTTPException
from services.agents_service import generate_assessment_from_resource

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

@router.post("/teacher/assessment-generation")
async def generate_assessment_api(data: dict):
    # The frontend sends { "resource_id": "...", "difficulty": "...", "count": 5 }
    resource_id = data.get("resource_id")
    
    if not resource_id:
        raise HTTPException(status_code=400, detail="No resource_id provided. I need a file to read!")
        
    try:
        # We call the 'Brain' service we tested earlier
        result = await generate_assessment_from_resource(resource_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"The AI Brain stumbled: {str(e)}")

# This keeps your old 'route' working just in case other parts of the app need it
@router.post("/route")
async def route_agent():
    return {"message": "Agent router is active. Use specific endpoints for generation."}
