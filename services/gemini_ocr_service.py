import base64
import io
import os
import uuid

from fastapi import HTTPException
from utils.logger import log_error, log_info

from services.ocr_service import MIN_REGIONS, PALETTE_SIZE, _line_confidence, _split_markdown

GEMINI_MODEL = "gemini-3-flash-preview"

TRANSCRIPTION_PROMPT = (
    "Please transcribe the handwritten text in this image exactly as it is written. "
    "Maintain the original structure, bullet points, numbered lists, and any layout features. "
    "Use Markdown headers (## Section Name) to separate visually distinct sections or topics. "
    "Do not add any interpretations, corrections, or summaries — just the raw transcription."
)


def _pdf_to_jpeg_pages(file_path: str) -> list[tuple[bytes, int, int]]:
    """Convert every PDF page to JPEG bytes using pypdfium2. Returns (jpeg_bytes, w, h)."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_path)
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=2.0)
            pil_img = bitmap.to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            pages.append((buf.getvalue(), pil_img.width, pil_img.height))
        return pages
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF → image conversion failed: {exc}")


def _call_gemini(client, img_bytes: bytes, img_mime: str) -> str:
    """
    Mirror the JS transcriber exactly: base64-encode the raw image bytes and
    send as inlineData inside a parts array — no PIL conversion, image sent as-is.
    """
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": img_mime, "data": base64_data}},
                        {"text": TRANSCRIPTION_PROMPT},
                    ]
                }
            ],
        )
        text = response.text or ""
        log_info("Gemini raw response", response=text)
        return text
    except Exception as exc:
        log_error("Gemini vision API error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Gemini vision call failed: {exc}")


def _build_regions(markdown: str, page_idx: int) -> list:
    """Return the full transcription as one region covering the whole page."""
    lines = [
        {"text": line, "confidence": _line_confidence(line)}
        for line in markdown.splitlines()
        if line.strip()
    ]
    return [{
        "id": f"r{page_idx}_0_{uuid.uuid4().hex[:6]}",
        "label": "Transcription",
        "colorIdx": 0,
        "bounds": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
        "lines": lines or [{"text": "", "confidence": 1.0}],
        "corrected": False,
    }]


async def perform_gemini_ocr(file_path: str, mime_type: str = "image/jpeg") -> dict:
    """
    Transcribe a local image or PDF using Gemini vision.

    Returns the same shape as perform_ocr() so the router and frontend need no changes:
        { "pages": [{ "page_number", "width", "height", "raw_markdown", "regions" }] }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    try:
        from google import genai
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-genai package is not installed. Run: pip install google-genai",
        )

    client = genai.Client(api_key=api_key)
    result_pages = []

    if mime_type == "application/pdf":
        log_info("Converting PDF to images for Gemini OCR", file=file_path)
        pdf_pages = _pdf_to_jpeg_pages(file_path)

        for page_num, (img_bytes, w, h) in enumerate(pdf_pages, start=1):
            log_info("Transcribing PDF page with Gemini", page=page_num)
            try:
                markdown = _call_gemini(client, img_bytes, "image/jpeg")
            except HTTPException as exc:
                log_error("Gemini vision failed on PDF page", page=page_num, detail=exc.detail)
                markdown = ""

            result_pages.append({
                "page_number": page_num,
                "width": w,
                "height": h,
                "raw_markdown": markdown,
                "regions": _build_regions(markdown, page_num),
            })

    else:
        log_info("Transcribing image with Gemini vision", file=file_path, mime=mime_type)
        with open(file_path, "rb") as f:
            img_bytes = f.read()

        # Get image dimensions for the page metadata
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            w, h = pil_img.width, pil_img.height
        except Exception:
            w, h = 0, 0

        markdown = _call_gemini(client, img_bytes, mime_type)

        result_pages.append({
            "page_number": 1,
            "width": w,
            "height": h,
            "raw_markdown": markdown,
            "regions": _build_regions(markdown, 1),
        })

    log_info("Gemini OCR complete", pages=len(result_pages))
    return {"pages": result_pages}
