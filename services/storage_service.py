import os
import aiofiles
from uuid import uuid4
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_upload(file: UploadFile) -> str:
    # We give it a unique ID so if two teachers upload "test.pdf", they don't overwrite each other.
    file_id = str(uuid4())
    extension = os.path.splitext(file.filename)[1]
    file_path = f"{UPLOAD_DIR}/{file_id}{extension}"
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
        
    return file_path
