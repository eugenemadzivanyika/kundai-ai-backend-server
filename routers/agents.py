import os
import uuid
import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from services.agents_service import generate_assessment_from_resource
from services.llm_service import call_llm
from services.asag_service import parse_json_from_ai

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

@router.post("/teacher/assessment-generation")
async def generate_assessment_api(data: dict):
    resource_id = data.get("resource_id")
    if not resource_id:
        raise HTTPException(status_code=400, detail="No resource_id provided. I need a file to read!")
    try:
        result = await generate_assessment_from_resource(resource_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"The AI Brain stumbled: {str(e)}")

@router.post("/student/assessment")
async def assess_student_submission(
    file: UploadFile = File(None),
    text: str = Form(None),
    module: str = Form("General"),
):
    """
    Accepts a student file upload (PDF/image) or raw text and returns a structured
    assessment result compatible with the frontend externalAssessmentService shape.
    """
    extracted_text = ""
    file_id = None
    content_type = "text"

    if file and file.filename:
        # Save the uploaded file temporarily so OCR can read it
        ext = os.path.splitext(file.filename)[1].lower()
        file_id = f"{uuid.uuid4()}{ext}"
        file_path = f"uploads/{file_id}"
        try:
            contents = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)

            # Extract text via pdfplumber (PDF) or pytesseract (image)
            if ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    extracted_text = "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    ).strip()
            elif ext in (".png", ".jpg", ".jpeg"):
                import pytesseract
                from PIL import Image
                img = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(img).strip()
            else:
                extracted_text = ""

            content_type = ext.lstrip(".")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
    elif text:
        extracted_text = text
        content_type = "text"
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or text content.")

    if not extracted_text:
        raise HTTPException(status_code=422, detail="Could not extract any text from the submission.")

    system_prompt = """You are a ZIMSEC Mathematics examiner.
Assess the student's submitted work and return a JSON object with this EXACT structure — no extra keys, no markdown fences:
{
  "is_correct_module": true,
  "confidence_assessment_score": 0.85,
  "total_possible_marks": 10,
  "marks_achieved": 7,
  "marks_percentage": 70,
  "overall_feedback": "One or two sentences of honest, encouraging feedback.",
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "criteria": [
    {"criterion": "Criterion name", "score": 3, "feedback": "Brief note"}
  ],
  "assessment_details": {
    "Q1": {"max_marks": 5, "awarded_marks": 3, "feedback": "Partial credit.", "improvement": "Show full working."}
  },
  "detected_module": "Mathematics",
  "mark_consistency_check": "Marks are consistent with the rubric.",
  "marking_scheme_used": false
}"""

    user_prompt = f"""Module: {module}

Student submission:
{extracted_text[:3000]}

Assess this work. Award marks out of 10. Be fair and specific."""

    try:
        response = await call_llm(user_prompt, system_content=system_prompt)
        result = parse_json_from_ai(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI assessment failed: {str(e)}")

    return {
        "module": module,
        "filename": file.filename if file else None,
        "content_type": content_type,
        "file_id": file_id,
        "assessment": result,
    }

# This keeps your old 'route' working just in case other parts of the app need it
@router.post("/route")
async def route_agent():
    return {"message": "Agent router is active. Use specific endpoints for generation."}
