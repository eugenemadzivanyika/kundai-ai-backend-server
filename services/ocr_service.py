import base64
import os
import re
import uuid

import httpx
from fastapi import HTTPException
from utils.logger import log_error, log_info

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
PALETTE_SIZE = 6
MIN_REGIONS = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _b64_encode(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_document(file_path: str, mime_type: str) -> dict:
    """Build the Mistral OCR document payload from a local file."""
    data = _b64_encode(file_path)
    data_url = f"data:{mime_type};base64,{data}"

    if mime_type.startswith("image/"):
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
    else:
        # PDF and other documents
        return {
            "type": "document_url",
            "document_url": data_url,
        }


# ── Markdown → region conversion ───────────────────────────────────────────────

def _split_markdown(md: str) -> list[tuple[str, list[str]]]:
    """
    Split a Markdown string into [(label, [lines])].
    Tries header-based sections first, falls back to paragraph splitting.
    """
    sections: list[tuple[str, list[str]]] = []

    header_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    headers = list(header_re.finditer(md))

    if headers:
        for i, match in enumerate(headers):
            label = re.sub(r"[*`_\\]", "", match.group(2)).strip()[:60]
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
            body = md[start:end].strip()
            lines = [l for l in body.splitlines() if l.strip()]
            if lines:
                sections.append((label or f"Section {i + 1}", lines))
        return sections or [("Content", [l for l in md.splitlines() if l.strip()])]

    # Fall back: paragraph splitting (double newline)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", md) if p.strip()]
    if not paragraphs:
        return [("Content", [l for l in md.splitlines() if l.strip()])]

    for i, para in enumerate(paragraphs):
        lines = [l for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        # First line (stripped of markdown decoration) becomes the label
        label = re.sub(r"[#*`_>|\\]", "", lines[0])[:50].strip() or f"Section {i + 1}"
        sections.append((label, lines))

    return sections


def _line_confidence(text: str) -> float:
    """Heuristic confidence for a single extracted line."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    # Very short or single-character fragments are less reliable
    if len(stripped) <= 2:
        return 0.60
    if len(stripped) <= 5:
        return 0.75
    # Lines that are purely numeric or symbol-heavy get moderate confidence
    if re.fullmatch(r"[\d\s\.\,\-\+\=\(\)\/]+", stripped):
        return 0.82
    return 0.95


def _markdown_to_regions(
    markdown: str,
    images: list,
    dimensions: dict,
    page_idx: int,
) -> list:
    """
    Convert a single Mistral OCR page into a list of OcrRegion dicts.

    Image regions get exact pixel bounding boxes (converted to %).
    Text regions are distributed vertically across remaining page space.
    """
    img_w = dimensions.get("width") or 1
    img_h = dimensions.get("height") or 1
    regions: list = []

    # ── Image regions (exact bounding boxes from Mistral) ────────────────────
    occupied: list[tuple[float, float]] = []  # (y_start%, y_end%)

    for img in images:
        tlx = img.get("top_left_x", 0)
        tly = img.get("top_left_y", 0)
        brx = img.get("bottom_right_x", img_w)
        bry = img.get("bottom_right_y", img_h)

        x_pct = round(tlx / img_w * 100, 1)
        y_pct = round(tly / img_h * 100, 1)
        w_pct = round((brx - tlx) / img_w * 100, 1)
        h_pct = round((bry - tly) / img_h * 100, 1)

        occupied.append((y_pct, y_pct + h_pct))

        regions.append({
            "id": f"r{page_idx}_img_{img.get('id', uuid.uuid4().hex[:6])}",
            "label": "Figure / Diagram",
            "colorIdx": len(regions) % PALETTE_SIZE,
            "bounds": {"x": x_pct, "y": y_pct, "w": w_pct, "h": h_pct},
            "lines": [{"text": f"[Embedded image: {img.get('id', 'figure')}]", "confidence": 1.0}],
            "corrected": False,
        })

    # ── Text regions (vertically distributed) ────────────────────────────────
    sections = _split_markdown(markdown)

    # Guarantee at least MIN_REGIONS sections so the overlay always covers the page
    while len(sections) < MIN_REGIONS:
        sections.append((f"Section {len(sections) + 1}", []))

    n = len(sections)

    # Divide available vertical space evenly; then nudge to avoid image overlaps
    step = 100.0 / n

    for i, (label, lines) in enumerate(sections):
        y_start = round(i * step, 1)
        y_end = round(min((i + 1) * step, 100.0), 1)

        # Nudge down past any overlapping image region
        for occ_y0, occ_y1 in occupied:
            if y_start < occ_y1 and y_end > occ_y0:
                y_start = occ_y1
                y_end = min(occ_y1 + step, 100.0)

        region_lines = []
        for raw_line in lines:
            text = raw_line.strip()
            if not text:
                continue
            region_lines.append({
                "text": text,
                "confidence": _line_confidence(text),
            })

        regions.append({
            "id": f"r{page_idx}_{i}_{uuid.uuid4().hex[:6]}",
            "label": label,
            "colorIdx": len(regions) % PALETTE_SIZE,
            "bounds": {
                "x": 0.0,
                "y": y_start,
                "w": 100.0,
                "h": max(round(y_end - y_start, 1), 4.0),
            },
            "lines": region_lines,
            "corrected": False,
        })

    return regions


# ── Main entry point ───────────────────────────────────────────────────────────

async def perform_ocr(file_path: str, mime_type: str = "image/png") -> dict:
    """
    Run Mistral OCR on a local file and return structured page data.

    Response shape:
    {
        "pages": [
            {
                "page_number": int,
                "width": int,
                "height": int,
                "raw_markdown": str,
                "regions": [
                    {
                        "id": str,
                        "label": str,
                        "colorIdx": int,
                        "bounds": {"x": float, "y": float, "w": float, "h": float},
                        "lines": [{"text": str, "confidence": float}],
                        "corrected": false
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured.")

    document = _build_document(file_path, mime_type)
    payload = {
        "model": MISTRAL_OCR_MODEL,
        "document": document,
        "include_image_base64": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log_info("Calling Mistral OCR", model=MISTRAL_OCR_MODEL, mime=mime_type)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(MISTRAL_OCR_URL, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:400]
            log_error("Mistral OCR HTTP error", status=exc.response.status_code, body=body)
            raise HTTPException(
                status_code=502,
                detail=f"Mistral OCR returned {exc.response.status_code}: {body}",
            )
        except Exception as exc:
            log_error("Mistral OCR request failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"OCR request failed: {exc}")

    data = resp.json()
    mistral_pages = data.get("pages", [])

    result_pages = []
    for page in mistral_pages:
        idx = page.get("index", len(result_pages) + 1)
        markdown = page.get("markdown", "")
        images = page.get("images", [])
        dimensions = page.get("dimensions", {})

        regions = _markdown_to_regions(markdown, images, dimensions, idx)

        result_pages.append({
            "page_number": idx,
            "width": dimensions.get("width", 0),
            "height": dimensions.get("height", 0),
            "raw_markdown": markdown,
            "regions": regions,
        })

    log_info("OCR complete", pages=len(result_pages))
    return {"pages": result_pages}
