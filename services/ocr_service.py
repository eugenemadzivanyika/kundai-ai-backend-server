import pytesseract
from PIL import Image
import pdfplumber
import os


async def perform_ocr(file_path: str) -> str:
    # 1. Identify what we are looking at
    if isinstance(payload, dict):
        file_path = f"uploads/{payload.get('file_id')}"
    else:
        file_path = payload
    
    extension = os.path.splitext(file_path)[1].lower()
    try:
        if extension == ".pdf":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text.strip()

        elif extension in [".png", ".jpg", ".jpeg"]:
            # Use Pillow to open the image
            img = Image.open(file_path)
            # Use Tesseract to 'read' the text
            text = pytesseract.image_to_string(img)
            return text.strip()
        
        return "Unsupported format for OCR."
    except Exception as e:
        return f"OCR Error: {str(e)}"
