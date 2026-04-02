from fastapi import APIRouter, UploadFile, File, HTTPException
from services.storage_service import save_upload

router = APIRouter(prefix="/resources", tags=["Resources"])

@router.post("/upload")
async def upload_resource(file: UploadFile = File(...)):
    # Validate it's a file we actually want
    if not file.content_type in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only PDFs and Images (JPG/PNG) are allowed!")
    
    path = await save_upload(file)
    return {"id": path.split('/')[-1], "url": f"/{path}", "name": file.filename}
