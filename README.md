# zivAI AI Services Backend (FastAPI)

This service will host AI/ML endpoints for zivAI (question generation, grading, OCR orchestration, recommendations). It is scaffolded with FastAPI and stub routes to be implemented once models are ready.

## Project structure
- `main.py` — central FastAPI app, router registration
- `routers/` — API route modules
- `services/` — business logic functions used by routers
- `requirements.txt` — Python dependencies
- `.env.example` — example port/CORS configuration

## Dedicated port
Default port for this service is **8001**. You can override it with:
```
AI_SERVICE_PORT=8001
```

## CORS configuration
Allowed origins are controlled via `ALLOWED_ORIGINS` (comma-separated).
Example:
```
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## API documentation (Swagger / OpenAPI)
FastAPI auto-generates API docs:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`
- **OpenAPI JSON**: `http://localhost:8001/openapi.json`

## Setup (local)
Prerequisites:
- Python 3.10+

1) **Create a virtual environment**
```bash
cd ai-services-backend
python3 -m venv .venv
```

2) **Activate the virtual environment**
```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

4) **Run the service**
```bash
# Linux/macOS
AI_SERVICE_PORT=8001 uvicorn main:app --reload --host 0.0.0.0 --port ${AI_SERVICE_PORT}
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Windows PowerShell
$env:AI_SERVICE_PORT=8001
uvicorn main:app --reload --host 0.0.0.0 --port $env:AI_SERVICE_PORT
```

## Health check
- `GET /health`

Example:
```bash
curl http://localhost:8001/health
```

Response:
```json
{"status":"ok","service":"ai-services-backend"}
```

## Service modules (stubs)
These are placeholder modules to be implemented when models are available:

- **OCR**
  - Router: `routers/ocr.py`
  - Service: `services/ocr_service.py`
  - Endpoint: `POST /ocr/extract`

- **ASAG (Automated Short Answer Grading)**
  - Router: `routers/asag.py`
  - Service: `services/asag_service.py`
  - Endpoint: `POST /asag/grade`

- **DKT (Deep Knowledge Tracing)**
  - Router: `routers/dkt.py`
  - Service: `services/dkt_service.py`
  - Endpoint: `POST /dkt/update`

- **Agents Router**
  - Router: `routers/agents.py`
  - Service: `services/agents_service.py`
  - Endpoint: `POST /agents/route`

- **Recommendations**
  - Router: `routers/recommendations.py`
  - Service: `services/recommendations_service.py`
  - Endpoint: `POST /recommendations/generate`

- **Content Generation** (lesson plans, notes, question variants)
  - Router: `routers/content.py`
  - Service: `services/content_service.py`
  - Endpoint: `POST /content/generate`

## Next steps
- Replace stub service logic with real model calls.
- Add request/response schemas (Pydantic models).
- Wire model configs and credentials via environment variables.
