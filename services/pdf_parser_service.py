from __future__ import annotations
import asyncio

# Minimum character count before we conclude the PDF is image-only (scanned).
# A typical one-page text PDF yields 2 000+ chars; scanned pages yield near-zero.
_MIN_TEXT_CHARS = 200


async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF, falling back to Gemini OCR for scanned documents."""
    text = await asyncio.to_thread(_extract_sync, file_path)

    if len(text.strip()) < _MIN_TEXT_CHARS:
        text = await _ocr_fallback(file_path)

    return text


def _extract_sync(file_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)


async def _ocr_fallback(file_path: str) -> str:
    """Use Gemini vision OCR to extract text from a scanned PDF."""
    try:
        from services.gemini_ocr_service import perform_gemini_ocr
        result = await perform_gemini_ocr(file_path, mime_type="application/pdf")
        pages = result.get("pages", [])
        return "\n\n".join(p.get("raw_markdown", "") for p in pages)
    except Exception as exc:
        # OCR is best-effort; return empty string so the caller can handle it
        import logging
        logging.getLogger(__name__).warning("Gemini OCR fallback failed: %s", exc)
        return ""
