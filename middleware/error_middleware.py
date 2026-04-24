import json
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import log_error


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Cache the body now so route handlers can still read it later
        await request.body()

        try:
            return await call_next(request)
        except Exception as exc:
            body = await _safe_parse_body(request)
            log_error(
                f"{request.method} {request.url.path} - {exc}",
                status_code=500,
                body=body,
                stack=traceback.format_exc(),
            )
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {exc}"},
            )


async def _safe_parse_body(request: Request) -> dict | str | None:
    content_type = request.headers.get("content-type", "")
    raw = await request.body()
    if not raw:
        return None
    if "application/json" in content_type:
        try:
            return json.loads(raw)
        except Exception:
            pass
    if "text/" in content_type:
        return raw.decode("utf-8", errors="replace")
    return f"<{len(raw)} bytes, content-type: {content_type}>"
