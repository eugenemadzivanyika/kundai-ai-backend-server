import asyncio
import base64
import io
import os
import re
import uuid
from fastapi import HTTPException
from utils.logger import log_error, log_info
from services.ocr_service import _line_confidence

GEMINI_MODEL = "gemini-3-flash-preview"

# --- UPDATED PROMPT INSTRUCTIONS ---
_SHARED_RULES = """\
You are a forensic transcription tool for handwritten mathematics assignments.
Reproduce exactly what is physically written — faithful to the student's own work.

OUTPUT FORMAT — you must use these conventions:
- Wrap every mathematical expression in LaTeX delimiters:
    • $...$ for inline math  (e.g. $x = 5$, $x \\geq 3$, $4a^2 - 5b$)
    • $$...$$ for a display equation on its own line
- Recognise common handwritten maths symbols and render them in LaTeX:
    • A fraction (number / expression over a rule line) → \\frac{top}{bottom}
    • ≥ or the sign that looks like > with an underline → \\geq
    • ≤ or the sign that looks like < with an underline → \\leq
    • × → \\times   |  · → \\cdot   |  ÷ → \\div
    • ∪ → \\cup      |  ∩ → \\cap    |  \\ (set difference) → \\setminus
    • Exponents written as superscripts → ^ (e.g. a^2)
    • Square root sign → \\sqrt{...}
    • ≠ → \\ne       |  ≈ → \\approx
- LAYOUT QUIRKS: If a number is deliberately circled in the working (like in HCF/LCM problems), indicate that by placing the number in parentheses and adding a brief italicized note if necessary.
- Preserve the student's own question numbering and layout using indentation and blank lines.


STRICT RULES — do not break these:
- NO ADDITIONS: Do NOT add any conversational filler, missing steps, commentary, labels, or text that you do not explicitly see in the images. Transcribe *only* what is there.
- DO NOT CORRECT: Do NOT correct the student's mathematical errors — if their algebra is wrong, transcribe it wrong.
- DO NOT REORDER: Maintain exact numerical order of the questions. Do NOT reorder or restructure anything.
- Output ONLY the transcription. No preamble, no sign-off.\
"""

TRANSCRIPTION_PROMPT = _SHARED_RULES

BATCH_PROMPT = (
    _SHARED_RULES
    + "\n\n"
    "You are given multiple images, each being one page of the same assignment. "
    "Combine the context and transcribe each page in chronological order, following the numbering from the student.\n\n"
    "Begin each page's transcription with ===PAGE_N=== on its own line "
    "(N is the 1-based page number).\n"
    "Example:\n===PAGE_1===\n<transcription>\n===PAGE_2===\n<transcription>"
)
# -----------------------------------

MAX_IMAGE_SIDE = 1600  # px — beyond this offers no OCR quality gain but balloons base64 payload


def _resize_for_gemini(img_bytes: bytes) -> tuple[bytes, str]:
    """
    Resize an image so its longest side is at most MAX_IMAGE_SIDE pixels, then
    re-encode as JPEG.  Returns (resized_bytes, "image/jpeg").
    No-ops (returns original bytes) if PIL is unavailable or the image is already small.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return img_bytes, "image/jpeg"


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


def _pdf_bytes_to_jpeg_pages(content: bytes) -> list[tuple[bytes, int, int]]:
    """Same as _pdf_to_jpeg_pages but accepts raw bytes instead of a file path."""
    try:
        import pypdfium2 as pdfium

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


async def _call_gemini(client, img_bytes: bytes, img_mime: str) -> str:
    """Send a single image to Gemini asynchronously and return the transcription."""
    img_bytes, img_mime = _resize_for_gemini(img_bytes)
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[{
                "parts": [
                    {"inline_data": {"mime_type": img_mime, "data": base64_data}},
                    {"text": TRANSCRIPTION_PROMPT},
                ]
            }],
        )
        text = response.text or ""
        log_info("Gemini raw response", response=text)
        return text
    except Exception as exc:
        log_error("Gemini vision API error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Gemini vision call failed: {exc}")


def _parse_batch_response(text: str, n: int) -> list[str]:
    """Split a batch response on ===PAGE_N=== delimiters into per-page transcriptions."""
    segments = re.split(r"===PAGE_\d+===", text)
    results = [s.strip() for s in segments if s.strip()]
    while len(results) < n:
        results.append("")
    return results[:n]


async def _call_gemini_batch(client, pages: list[tuple[bytes, str]]) -> list[str]:
    """Send all pages in a single Gemini call. Returns per-page transcription strings."""
    resized = [_resize_for_gemini(img) for img, _ in pages]
    parts = [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(img).decode()}}
        for img, mime in resized
    ]
    parts.append({"text": BATCH_PROMPT})

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[{"parts": parts}],
        )
        text = response.text or ""
        log_info("Gemini batch response", pages=len(pages), chars=len(text))
        return _parse_batch_response(text, len(pages))
    except Exception as exc:
        log_error("Gemini batch vision API error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Gemini batch vision call failed: {exc}")


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

    - Single image: one async Gemini call.
    - PDF: all pages sent in a single batch Gemini call, parsed by ===PAGE_N=== delimiters.

    Returns the same shape as perform_ocr():
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
        log_info("Converting PDF to images for Gemini batch OCR", file=file_path)
        pdf_pages = _pdf_to_jpeg_pages(file_path)

        page_inputs = [(img_bytes, "image/jpeg") for img_bytes, _, _ in pdf_pages]
        log_info("Sending all PDF pages in one Gemini batch call", pages=len(page_inputs))
        transcriptions = await _call_gemini_batch(client, page_inputs)

        for page_num, ((img_bytes, w, h), markdown) in enumerate(
            zip(pdf_pages, transcriptions), start=1
        ):
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

        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            w, h = pil_img.width, pil_img.height
        except Exception:
            w, h = 0, 0

        markdown = await _call_gemini(client, img_bytes, mime_type)

        result_pages.append({
            "page_number": 1,
            "width": w,
            "height": h,
            "raw_markdown": markdown,
            "regions": _build_regions(markdown, 1),
        })

    log_info("Gemini OCR complete", pages=len(result_pages))
    return {"pages": result_pages}


async def perform_gemini_ocr_batch(
    upload_files: list[tuple[bytes, str, str]],
) -> dict:
    """
    Transcribe multiple files (images or PDFs) in a single Gemini batch call.

    Args:
        upload_files: list of (content_bytes, mime_type, filename)

    Returns:
        { "files": [{ "pages": [BackendOcrPage, ...] }, ...] }
        One entry per input file, in the same order.
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

    # Expand every input file into individual page images.
    # Track (image_bytes, mime, file_index, page_within_file, width, height).
    all_pages: list[tuple[bytes, str, int, int, int, int]] = []

    for fi, (content, mime, _) in enumerate(upload_files):
        if mime == "application/pdf":
            pdf_pages = _pdf_bytes_to_jpeg_pages(content)
            for pi, (img_bytes, w, h) in enumerate(pdf_pages):
                all_pages.append((img_bytes, "image/jpeg", fi, pi, w, h))
        else:
            w, h = 0, 0
            try:
                from PIL import Image as PILImage
                pil = PILImage.open(io.BytesIO(content))
                w, h = pil.width, pil.height
            except Exception:
                pass
            all_pages.append((content, mime, fi, 0, w, h))

    log_info("Sending all pages in one Gemini batch call", total_pages=len(all_pages), files=len(upload_files))

    # One Gemini call for everything.
    page_inputs = [(img, mime) for img, mime, _, _, _, _ in all_pages]
    transcriptions = await _call_gemini_batch(client, page_inputs)

    # Reassemble results grouped by original file index.
    file_results: list[list] = [[] for _ in upload_files]
    for i, (_, _, fi, within_pi, w, h) in enumerate(all_pages):
        markdown = transcriptions[i] if i < len(transcriptions) else ""
        file_results[fi].append({
            "page_number": within_pi + 1,
            "width": w,
            "height": h,
            "raw_markdown": markdown,
            "regions": _build_regions(markdown, within_pi),
        })

    log_info("Gemini batch OCR complete", files=len(file_results))
    return {"files": [{"pages": pages} for pages in file_results]}