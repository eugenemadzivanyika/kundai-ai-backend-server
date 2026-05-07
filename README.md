# KundAI AI Services Backend

Python / FastAPI service providing AI and ML endpoints for the KundAI platform. Runs independently of the main Node.js backend and is called directly by the React frontend and the Node.js server.

## Default port: 8000

---

## Project structure

```
kundai-ai-services-backend/
├── main.py                  # FastAPI app entry point, router registration
├── requirements.txt
├── routers/                 # Route modules
│   ├── health.py
│   ├── ocr.py
│   ├── asag.py
│   ├── bkt.py
│   ├── agents.py
│   ├── developmentPlan.py
│   ├── devplan_content_generation.py
│   ├── assessment_generation.py
│   ├── ai_tutor.py
│   ├── content.py
│   └── resources.py
├── services/                # Business logic used by routers
│   ├── gemini_ocr_service.py
│   ├── ocr_service.py
│   ├── asag_service.py
│   ├── bkt_service.py
│   ├── agents_service.py
│   ├── developmentPlan_service.py
│   ├── devPlan_content_generation_service.py
│   ├── assessment_generation_service.py
│   ├── ai_tutor_service.py
│   ├── content_service.py
│   ├── llm_service.py
│   ├── storage_service.py
│   └── health_service.py
├── middleware/
│   └── error_middleware.py
├── utils/
│   └── logger.py
└── uploads/                 # Temp storage for uploaded files during processing
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/ocr/extract` | OCR on uploaded image or PDF (Gemini Vision + Tesseract) |
| `POST` | `/asag/grade` | Automated Short Answer Grading |
| `POST` | `/bkt/update` | Bayesian Knowledge Tracing — update student knowledge state |
| `POST` | `/development-plan/generate` | Generate a personalised student development plan |
| `POST` | `/devplan-content/generate` | Generate content for development plan steps |
| `POST` | `/assessment/generate` | Generate assessment questions from syllabus topics |
| `POST` | `/ai-tutor/chat` | AI tutor conversational endpoint |
| `POST` | `/agents/route` | Multi-agent routing |
| `POST` | `/content/generate` | Lesson notes and study content generation |
| `GET` | `/resources` | Resource listing |

Static uploads served at `/uploads`.

---

## Setup

**Prerequisites**
- Python 3.10+
- Tesseract OCR installed on the system:
  ```bash
  sudo apt install tesseract-ocr        # Ubuntu/Debian
  brew install tesseract                # macOS
  ```
- Google Gemini API key

**Install**
```bash
cd kundai-ai-services-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment variables** — create a `.env` file:
```env
GEMINI_API_KEY=your-gemini-api-key
AI_SERVICE_PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

**Run (development)**
```bash
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## API documentation

FastAPI auto-generates interactive docs while the service is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "service": "kundai-ai-services-backend"}
```
