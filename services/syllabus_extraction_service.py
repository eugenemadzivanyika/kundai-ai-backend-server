import asyncio
import base64
import json
import os
import re
from fastapi import HTTPException
from utils.logger import log_error, log_info

GEMINI_MODEL = "gemini-3-flash-preview"

EXTRACTION_PROMPT = """\
You are an expert curriculum analyst for Zimbabwean secondary school education (ZIMSEC).
Analyse this syllabus document and extract EVERY individual learning topic as a curriculum attribute.

Return ONLY a JSON object with a single key "attributes" whose value is an array of objects.
Each object must have exactly these fields:

  "attribute_id"         — unique code following the pattern already used in the document
                           (e.g. "MATH-F1-RN-01", "MATH-F2-ALG-03"). If codes are not printed
                           in the syllabus, generate them as: SUBJECT-FLEVEL-UNITABBREV-SEQ
  "name"                 — short topic name (≤ 6 words)
  "parent_unit"          — the chapter/unit heading this topic belongs to
  "level"                — form level exactly as "Form 1", "Form 2", etc.
  "description"          — one sentence starting with an action verb (e.g. "Calculate…", "Identify…")
  "prerequisites"        — array of attribute_id strings (can be [])
  "total_mastery_points" — always the number 100
  "tags"                 — array containing: exam board (e.g. "ZIMSEC"), paper type
                           ("Paper 1" or "Paper 2"), and skill type
                           ("Calculations", "Problem Solving", "Graphs", etc.)

No markdown fences, no preamble, no trailing text — return raw JSON only.
"""


def _pdf_bytes_to_images(content: bytes) -> list[tuple[bytes, int, int]]:
    """Convert PDF bytes to JPEG page images using pypdfium2."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image as PILImage
        import io

        pdf = pdfium.PdfDocument(content)
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


def _resize(img_bytes: bytes) -> tuple[bytes, str]:
    """Resize image so longest side ≤ 1600 px and re-encode as JPEG."""
    try:
        from PIL import Image as PILImage
        import io

        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > 1600:
            scale = 1600 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return img_bytes, "image/jpeg"


def _parse_attributes(raw: str) -> list[dict]:
    """Extract the attributes array from Gemini's response, tolerating minor formatting."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    try:
        obj = json.loads(clean)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("attributes", "topics", "items", "data"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError("No attributes array found in response")
    except json.JSONDecodeError as exc:
        log_error("JSON parse failed on Gemini extraction response", error=str(exc), raw=raw[:500])
        raise HTTPException(status_code=502, detail=f"AI returned malformed JSON: {exc}")


async def extract_attributes_from_syllabus(
    file_content_b64: str,
    mime_type: str,
) -> list[dict]:
    """
    Send a syllabus file (base64-encoded) to Gemini and extract curriculum attributes.

    Args:
        file_content_b64: base64-encoded file bytes
        mime_type:        MIME type of the file (e.g. "application/pdf")

    Returns:
        List of attribute dicts matching the CourseAttribute schema.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    try:
        from google import genai
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-genai package not installed. Run: pip install google-genai",
        )

    client = genai.Client(api_key=api_key)
    file_bytes = base64.b64decode(file_content_b64)

    if mime_type == "application/pdf":
        log_info("Converting syllabus PDF to images for Gemini extraction")
        pdf_pages = _pdf_bytes_to_images(file_bytes)
        page_inputs: list[tuple[bytes, str]] = [
            _resize(img) for img, _, _ in pdf_pages
        ]
    else:
        page_inputs = [_resize(file_bytes)]

    log_info("Sending syllabus to Gemini for attribute extraction", pages=len(page_inputs))

    parts = [
        {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(img).decode(),
            }
        }
        for img, mime in page_inputs
    ]
    parts.append({"text": EXTRACTION_PROMPT})

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[{"parts": parts}],
        )
        raw = response.text or ""
        log_info("Gemini extraction response received", chars=len(raw))
    except Exception as exc:
        log_error("Gemini extraction API error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Gemini API call failed: {exc}")

    attributes = _parse_attributes(raw)
    log_info("Syllabus attributes extracted", count=len(attributes))
    return attributes
