# KundAI AI Services Backend

Python / FastAPI microservice providing all AI and ML capabilities for the KundAI adaptive learning platform. Runs independently of the main Node.js backend on **port 8000** and is called by both the React frontend and the Node.js server.

The service is purpose-built for the Zimbabwean secondary education context — ZIMSEC O-Level curriculum, localised examples (ZiG currency, Harare/Gweru/Mbare settings, names like Farai and Chipo), and direct alignment to syllabus attribute structures.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [AI & ML Stack](#ai--ml-stack)
- [Authentication](#authentication)
- [Endpoints Reference](#endpoints-reference)
- [Core Services](#core-services)
- [Middleware & Utilities](#middleware--utilities)
- [Setup & Running](#setup--running)
- [Environment Variables](#environment-variables)
- [Implementation Status](#implementation-status)

---

## Architecture Overview

```
React Frontend  ──┐
                  ├──▶  FastAPI (port 8000)  ──▶  Mistral API  (primary LLM + embeddings + OCR)
Node.js Backend ──┘          │                ──▶  Google Gemini  (vision OCR + LLM fallback)
                             │
                             ├──▶  ChromaDB  (persistent vector store, ./data/chromadb/)
                             └──▶  uploads/  (temporary file storage during requests)
```

**LLM Fallback Chain:** All LLM calls go to Mistral first (`mistral-small-latest`). On a 429 rate-limit response, the service automatically retries the same prompt against Google Gemini (`gemini-2.5-flash`).

**OCR Fallback Chain:** Gemini Vision is the primary OCR engine (`gemini-2.5-flash`). If no Gemini API key is configured, the service falls back to Mistral's OCR API.

**RAG pipeline:** Documents are chunked, embedded with Mistral's `mistral-embed` model, and stored in a per-subject ChromaDB collection. Retrieval uses L2 distance with a configurable relevance threshold (default 0.45).

All routes except `/health` require a valid JWT Bearer token.

---

## Directory Structure

```
kundai-ai-services-backend/
├── main.py                                   # App entry point, CORS config, router registration
├── requirements.txt                          # Python dependencies
├── .env                                      # API keys and secrets (not committed)
│
├── routers/                                  # HTTP route handlers (thin — delegate to services)
│   ├── health.py                             # GET /health
│   ├── ocr.py                                # POST /ocr/extract, /ocr/extract-batch
│   ├── asag.py                               # POST /asag/grade
│   ├── bkt.py                                # POST /bkt/update
│   ├── agents.py                             # POST /api/v1/agents/...
│   ├── developmentPlan.py                    # POST /developmentplan/generate-missions
│   ├── devplan_content_generation.py         # POST /devPlan-content-gen/...
│   ├── assessment_generation.py              # POST /assessment-gen/generate
│   ├── ai_tutor.py                           # POST /ai-tutor/introduce, /ai-tutor/coaching
│   ├── content.py                            # POST /content/generate  [STUB]
│   ├── resources.py                          # POST /resources/upload
│   ├── syllabus_extraction.py                # POST /syllabus-extraction/extract
│   ├── rag.py                                # POST /rag/inject, /rag/query, etc.
│   ├── notes.py                              # POST /notes/generate, /notes/chat
│   ├── persisted_notes.py                    # POST /persisted-notes/generate
│   └── reteach.py                            # POST /reteach/generate-cards
│
├── services/                                 # All business logic lives here
│   ├── llm_service.py                        # Unified LLM client (Mistral → Gemini fallback)
│   ├── gemini_ocr_service.py                 # Gemini Vision OCR (primary)
│   ├── ocr_service.py                        # Mistral OCR (fallback)
│   ├── asag_service.py                       # Grading pipeline + cognitive profiling
│   ├── bkt_service.py                        # Bayesian Knowledge Tracing formula
│   ├── agents_service.py                     # Teacher & student agent orchestration
│   ├── ai_tutor_service.py                   # Socratic tutoring + checkpoint validation
│   ├── notes_service.py                      # RAG-grounded note generation + topic chat
│   ├── assessment_generation_service.py      # ZIMSEC-aligned question generation
│   ├── developmentPlan_service.py            # 3-mission remediation path generation
│   ├── devPlan_content_generation_service.py # Theory, practice, quiz, challenge content
│   ├── reteach_service.py                    # Class intervention scripts + exit tickets
│   ├── rag_service.py                        # ChromaDB inject / query / coverage analysis
│   ├── syllabus_extraction_service.py        # Syllabus PDF → structured curriculum attributes
│   ├── pdf_parser_service.py                 # PDF text extraction (PyMuPDF → pypdf → OCR)
│   ├── document_classifier_service.py        # Classifies doc as question_paper or learning_material
│   ├── question_extractor_service.py         # Parses individual questions from exam papers
│   ├── storage_service.py                    # File upload to uploads/ with UUID naming
│   ├── health_service.py                     # Returns service status dict
│   ├── content_service.py                    # [STUB] — not yet implemented
│   └── __init__.py
│
├── middleware/
│   ├── auth.py                               # JWT validation (RS256 preferred, HS256 fallback)
│   ├── error_middleware.py                   # Global exception handler → structured 500 JSON
│   └── __init__.py
│
├── utils/
│   ├── logger.py                             # JSON logger: console (DEBUG+) + rotating file (ERROR+)
│   └── __init__.py
│
├── uploads/                                  # Temporary storage for uploaded files during processing
├── logs/                                     # Rotating error logs (5 MB per file, 3 backups)
└── data/chromadb/                            # Persistent ChromaDB vector database
```

---

## AI & ML Stack

| Component | Model / Library | Role |
|---|---|---|
| Mistral API | `mistral-small-latest` | Primary LLM for all generation tasks |
| Mistral API | `mistral-embed` | Text embeddings for RAG vector store |
| Mistral API | Mistral OCR | Fallback OCR when Gemini unavailable |
| Google Gemini | `gemini-2.5-flash` | Primary vision OCR; LLM fallback on Mistral 429 |
| ChromaDB | Persistent client | Vector database for curriculum RAG documents |
| PyMuPDF (`fitz`) | — | Fast PDF text extraction (primary) |
| pypdf | — | PDF text extraction fallback |
| pypdfium2 | — | PDF → JPEG page rendering for Gemini vision |
| pytesseract | System Tesseract | Image OCR of last resort |
| Pillow | — | Image resize and JPEG compression |
| pydantic | v2 | Request/response schema validation |
| PyJWT | — | JWT token decoding (RS256 / HS256) |

The project uses **Mistral and Google APIs exclusively** — no Anthropic/Claude SDK is used anywhere.

---

## Authentication

All routes except `GET /health` are protected by the `require_user` FastAPI dependency declared in `middleware/auth.py`.

Tokens must be sent as:
```
Authorization: Bearer <jwt>
```

**Algorithm preference:**
1. **RS256** — if `JWT_PUBLIC_KEY` is set in `.env` (X.509 PEM or newline-escaped certificate)
2. **HS256** — fallback using `JWT_SECRET`

The middleware returns `401` for expired or invalid tokens and `500` if neither key is configured.

---

## Endpoints Reference

### Health

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `GET` | `/health` | No | Complete | Returns `{"status": "ok", "service": "ai-services-backend"}` |

---

### OCR — Optical Character Recognition

Routes: `routers/ocr.py` → `services/gemini_ocr_service.py` / `services/ocr_service.py`

Gemini Vision is the primary OCR engine. Mistral OCR is used as a fallback if no Gemini API key is present. Both engines return region objects with bounding boxes, confidence scores, and LaTeX-formatted mathematics (inline `$...$`, display `$$...$$`).

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/ocr/extract` | Yes | Complete | Extract text from a single uploaded image or PDF |
| `POST` | `/ocr/extract-batch` | Yes | Complete | Process multiple files in one request; optional question-aware mapping that assigns student answer regions to exam `questionId` fields with `partAnswers` |

---

### ASAG — Automated Short Answer Grading

Routes: `routers/asag.py` → `services/asag_service.py`

Two-step pipeline:

1. **Grading pass** — for each question, produces a `chainOfThought` with awarded score (raw marks, never percentages), per-question feedback, and detected misconceptions.
2. **Cognitive profile pass** — maps grading output to syllabus attributes; returns `strengths`, `deficiencies` (each linked to an `attributeName`), and a `suggestedTutorApproach`.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/asag/grade` | Yes | Complete | Grade student answers and build a cognitive profile in a single call |

---

### BKT — Bayesian Knowledge Tracing

Routes: `routers/bkt.py` → `services/bkt_service.py`

Pure-formula update, no LLM involved. Implements the standard BKT equations:

```
# Posterior probability of knowledge given observation
if correct:
    P(known|obs) = (prior × (1 − p_slip)) / ((prior × (1 − p_slip)) + ((1 − prior) × p_guess))
else:
    P(known|obs) = (prior × p_slip) / ((prior × p_slip) + ((1 − prior) × (1 − p_guess)))

# Apply learning probability
new_mastery = P(known|obs) + (1 − P(known|obs)) × p_transit
```

Default parameters: `p_guess = 0.2`, `p_slip = 0.1`, `p_transit = 0.1`

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/bkt/update` | Yes | Complete | Update a student's mastery estimate for one knowledge component |

---

### Development Plan

Routes: `routers/developmentPlan.py` → `services/developmentPlan_service.py`

Generates a personalised 3-mission remediation path based on a student's cognitive profile.

| Mission | Focus | Step Types |
|---|---|---|
| Mission 1 — Knowledge | Bridges identified strengths into gap areas | Theory, Interactive_Exercise |
| Mission 2 — Practice | Targets specific misconceptions directly | Hinted_Practice, Interactive_Exercise |
| Mission 3 — Mastery | ZIMSEC-style final evaluation | Hinted_Practice, mastery check |

Each step has an `exitCheckpoint` — a question and `expectedLogic` string — that must be satisfied before the next step unlocks.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/developmentplan/generate-missions` | Yes | Complete | Generate a 3-mission development plan from a student cognitive profile |

---

### Development Plan Content Generation

Routes: `routers/devplan_content_generation.py` → `services/devPlan_content_generation_service.py`

Five distinct generators that produce the actual learning content for each step type within a mission.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/devPlan-content-gen/theory` | Yes | Complete | Personalised theory mini-module; leads with addressing known misconceptions before teaching new material |
| `POST` | `/devPlan-content-gen/practice` | Yes | Complete | Set of 5 hinted practice problems targeting specific deficiencies |
| `POST` | `/devPlan-content-gen/quiz` | Yes | Complete | 5-question ZIMSEC-style MCQ mastery check; distractors are engineered to expose identified misconceptions |
| `POST` | `/devPlan-content-gen/unit-challenge` | Yes | Complete | MCQ bank covering all attributes in a unit (student-facing practice) |
| `POST` | `/devPlan-content-gen/subject-challenge` | Yes | Complete | Personalised subject-wide challenge; 60% of questions target the bottom 40% of attributes by mastery score |

---

### AI Tutor

Routes: `routers/ai_tutor.py` → `services/ai_tutor_service.py`

Socratic conversational tutor personalised to a student's profile. Uses RAG to supplement responses with relevant curriculum documents. Formatting follows WhatsApp conventions (`*bold*`, `_italic_`, Unicode math — no LaTeX) since output targets a chat interface.

**Coaching turn logic:**
- **First turn (TEACH phase):** Hook → Feynman explanation → personalisation using the student's known strengths → worked example → checkpoint probe question
- **Subsequent turns (RESPOND & ADVANCE phase):** Direct answer to the student's reply → diagnosis → silent checkpoint assessment → advance or re-explain

The tutor silently evaluates each response against the step's `checkpointExpectedLogic` and returns `checkpointPassed: true/false` so the frontend knows when to unlock the next step without exposing the evaluation to the student.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/ai-tutor/introduce` | Yes | Complete | Generate a welcome message when a student is assigned a development plan |
| `POST` | `/ai-tutor/coaching` | Yes | Complete | Multi-turn Socratic coaching for a single step, with silent checkpoint validation |

---

### Agents

Routes: `routers/agents.py` → `services/agents_service.py`

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/api/v1/agents/teacher/assessment-generation` | Yes | Complete | Accept a teacher-uploaded resource (PDF or image), extract content, and generate a structured assessment |
| `POST` | `/api/v1/agents/student/assessment` | Yes | Complete | Accept a student file or text submission and return a structured assessment result JSON |
| `POST` | `/api/v1/agents/route` | Yes | **STUB** | Legacy agent router — currently returns a placeholder acknowledgement, no routing logic implemented |

---

### Assessment Generation

Routes: `routers/assessment_generation.py` → `services/assessment_generation_service.py`

Generates ZIMSEC-aligned exam questions from a list of syllabus attributes. Respects paper style constraints:

- **Paper 1** — short/direct questions; `multiple_choice` is forbidden, only `short_answer`
- **Paper 2** — multi-step structured questions
- **Difficulty levels:** easy (formative), medium (school test), hard (O-Level)
- **Question structure:** `id`, `stem`, `solution_schema` (step-by-step mark scheme), `final_answer`
- Supports LaTeX in question stems and solutions
- Includes truncation recovery: if the LLM response is cut off mid-JSON, the service salvages all complete questions

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/assessment-gen/generate` | Yes | Complete | Generate questions from syllabus attributes at a specified difficulty and count |

---

### Syllabus Extraction

Routes: `routers/syllabus_extraction.py` → `services/syllabus_extraction_service.py`

Parses a syllabus document (PDF or image) using Gemini Vision. PDFs are converted to JPEG pages via pypdfium2 at 2× scale and sent to Gemini in a single batch call. Returns a structured list of curriculum attributes following the ID convention `SUBJECT-FLEVEL-UNITABBREV-SEQ` (e.g. `MATH-F1-ALG-02`).

Each attribute includes: `attribute_id`, `name`, `parent_unit`, `level`, `description`, `prerequisites`, `tags`.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/syllabus-extraction/extract` | Yes | Complete | Parse a syllabus PDF or image into a structured attribute list |

---

### RAG — Retrieval-Augmented Generation

Routes: `routers/rag.py` → `services/rag_service.py`

Manages the curriculum knowledge base in ChromaDB. Documents are stored per-subject in a collection named `subject_{subject_id}`.

**Injection pipeline:**
1. Extract text from PDF/image (`pdf_parser_service`)
2. Classify document as `question_paper` or `learning_material` (`document_classifier_service`)
3. Chunk text (500-word chunks, 60-word overlap)
4. For question papers: additionally extract individual questions as discrete chunks (`question_extractor_service`)
5. Embed chunks via Mistral `mistral-embed`
6. Store in ChromaDB with metadata: `source_file`, `document_id`, `subject_id`, `document_type`, `chunk_index`

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/rag/inject` | Yes | Complete | Upload a PDF or image, parse, classify, embed, and store in the vector DB |
| `POST` | `/rag/query` | Yes | Complete | Semantic search — returns top-k chunks by L2 distance for a query string |
| `POST` | `/rag/coverage-query` | Yes | Complete | Check which curriculum attributes have relevant documents indexed; returns attribute IDs with `max_relevance` scores |
| `DELETE` | `/rag/documents/{document_id}` | Yes | Complete | Remove all chunks belonging to a document from the vector store |
| `GET` | `/rag/status/{subject_id}` | Yes | Complete | Return total chunk count and document list for a subject |

---

### Notes

Routes: `routers/notes.py` → `services/notes_service.py`

Generates Markdown revision notes grounded in RAG-retrieved curriculum documents. Notes are personalised based on the student's cognitive profile — stronger areas receive concise treatment, identified gaps receive deeper explanation with additional worked examples.

**Note structure:** Overview → Key Concepts → Worked Examples → Summary → Quick Recall Questions

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/notes/generate` | Yes | Complete | Generate structured Markdown revision notes for a topic, RAG-grounded and profile-personalised |
| `POST` | `/notes/chat` | Yes | Complete | Multi-turn Q&A while a student reads their notes; context window includes the current note content (truncated to 3000 chars) and RAG supplementary results |

---

### Persisted Notes

Routes: `routers/persisted_notes.py`

Same note generation as `/notes/generate` but additionally posts the generated Markdown back to the Node.js backend via HTTP callback so it is stored in the main database.

Callback target:
```
POST {NODE_API_URL}/api/subject-notes/{subject_id}/topics/{topic}/subtopics/{subtopic}
Body: { "content": "<markdown>" }
```

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/persisted-notes/generate` | Yes | Complete | Generate notes and persist them to the Node.js backend in one call |

---

### Reteach Cards

Routes: `routers/reteach.py` → `services/reteach_service.py`

Generates classroom intervention materials for a teacher who needs to re-teach a concept to a group of students with identified misconceptions.

Output per card:
- **Intervention script** — 150–250 word teacher-facing 2-minute re-teach script
- **Exit ticket** — 3 questions (at least 1 MCQ, at least 1 short answer) to verify class understanding after the intervention

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/reteach/generate-cards` | Yes | Complete | Generate intervention scripts and exit tickets for a class re-teach session |

---

### Resources

Routes: `routers/resources.py` → `services/storage_service.py`

Simple file upload handler. Saves files to `uploads/` with a UUID-based filename to prevent collisions. Accepted MIME types: `application/pdf`, `image/jpeg`, `image/png`.

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/resources/upload` | Yes | Complete | Upload a PDF or image; returns `file_id` and storage path |

---

### Content Generation

Routes: `routers/content.py` → `services/content_service.py`

| Method | Path | Auth | Status | Description |
|---|---|---|---|---|
| `POST` | `/content/generate` | Yes | **STUB** | Returns `{"status": "stub"}` — not yet implemented |

---

## Core Services

### `llm_service.py` — Unified LLM Client

Central function: `call_llm(user_content, system_content, json_mode=True, history=None)`

- Sends requests to the Mistral chat completion API
- On HTTP 429, transparently retries against Gemini (`gemini-2.5-flash`)
- `json_mode=True` instructs the LLM to return valid JSON (used by all structured-output endpoints)
- `history` injects prior conversation turns between the system prompt and the current user message, enabling multi-turn dialogue in the AI tutor and notes chat features

### `asag_service.py` — Grading & Cognitive Profiling

Contains a `parse_json_from_ai()` utility that handles three failure modes in LLM JSON output:
1. Markdown code fences (` ```json ... ``` `) — stripped before parsing
2. Truncated responses cut off mid-JSON — partial array/object recovery
3. Bare backslashes from LaTeX in JSON strings — escaped to `\\` before `json.loads()`

### `rag_service.py` — ChromaDB Vector Store

Collections are scoped per subject. The coverage query endpoint embeds each curriculum attribute name, queries ChromaDB, and returns any attribute whose nearest document falls within the relevance threshold (L2 distance ≤ 0.45 by default).

### `document_classifier_service.py` — Heuristic + LLM Classification

First applies regex heuristics (numbered question patterns, `[marks]` annotations, sub-part labels `a) b) c)`, action verbs like *Evaluate*, *Simplify*, *Solve*). If heuristics are inconclusive, sends the first 2000 characters to the LLM for classification. Returns `"question_paper"` or `"learning_material"`.

### `syllabus_extraction_service.py` — Vision-Based Syllabus Parser

Converts PDF pages to JPEG at 2× resolution using pypdfium2, then sends all pages in a single Gemini Vision request. This approach captures multi-page syllabus layouts (tables spanning pages, prerequisite chains) in one structured extraction pass.

### `devPlan_content_generation_service.py` — Subject Challenge Weighting

The subject challenge generator applies a deliberate weighting: it allocates 60% of questions to the bottom 40% of a student's attributes ranked by mastery score. This ensures remediation effort is concentrated on genuine weak points rather than spreading evenly across the subject.

---

## Middleware & Utilities

### Error Middleware (`middleware/error_middleware.py`)

`ErrorLoggingMiddleware` wraps every request. On any unhandled exception it logs the HTTP method, path, request body, error message, and full stack trace, then returns a structured `{"detail": "Internal server error"}` 500 JSON response.

### Logger (`utils/logger.py`)

Structured JSON logger with two handlers:
- **Console:** `DEBUG` level and above
- **File:** `ERROR` level and above → `logs/error.log` with rotating file handler (5 MB per file, 3 backup files)

Helper functions: `log_error(message, **meta)`, `log_info(message, **meta)`

---

## Setup & Running

**Prerequisites**
- Python 3.10+
- Tesseract OCR binary (used as last-resort fallback):
  ```bash
  sudo apt install tesseract-ocr        # Ubuntu/Debian
  brew install tesseract                # macOS
  ```

**Install**
```bash
cd kundai-ai-services-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run (development)**
```bash
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Run (production)**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

**Interactive API docs (while running)**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

**Health check**
```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "ai-services-backend"}
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required — Mistral API (primary LLM, embeddings, OCR fallback)
MISTRAL_API_KEY=your-mistral-api-key

# Required — Google Gemini (primary vision OCR, LLM fallback on Mistral 429)
GEMINI_API_KEY=your-gemini-api-key

# Required — JWT authentication
# Option A: RS256 asymmetric (preferred for production)
JWT_PUBLIC_KEY=-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----

# Option B: HS256 symmetric (development fallback)
JWT_SECRET=your-jwt-secret

# Optional — Node.js backend URL for persisted notes callback
NODE_API_URL=http://localhost:3000
```

CORS is currently hard-coded to `http://localhost:5173` and `http://127.0.0.1:5173`. To change allowed origins, update the `allow_origins` list in `main.py`.

---

## Implementation Status

| Router Module | Endpoint(s) | Status |
|---|---|---|
| `health.py` | `GET /health` | Complete |
| `ocr.py` | `POST /ocr/extract`, `POST /ocr/extract-batch` | Complete |
| `asag.py` | `POST /asag/grade` | Complete |
| `bkt.py` | `POST /bkt/update` | Complete |
| `developmentPlan.py` | `POST /developmentplan/generate-missions` | Complete |
| `devplan_content_generation.py` | `POST /devPlan-content-gen/theory`, `/practice`, `/quiz`, `/unit-challenge`, `/subject-challenge` | Complete |
| `ai_tutor.py` | `POST /ai-tutor/introduce`, `POST /ai-tutor/coaching` | Complete |
| `agents.py` | `POST /api/v1/agents/teacher/assessment-generation` | Complete |
| `agents.py` | `POST /api/v1/agents/student/assessment` | Complete |
| `agents.py` | `POST /api/v1/agents/route` | **Stub** — placeholder response only |
| `assessment_generation.py` | `POST /assessment-gen/generate` | Complete |
| `syllabus_extraction.py` | `POST /syllabus-extraction/extract` | Complete |
| `rag.py` | `POST /rag/inject`, `/query`, `/coverage-query`, `DELETE /rag/documents/{id}`, `GET /rag/status/{id}` | Complete |
| `notes.py` | `POST /notes/generate`, `POST /notes/chat` | Complete |
| `persisted_notes.py` | `POST /persisted-notes/generate` | Complete |
| `reteach.py` | `POST /reteach/generate-cards` | Complete |
| `resources.py` | `POST /resources/upload` | Complete |
| `content.py` | `POST /content/generate` | **Stub** — returns `{"status": "stub"}` |

**28 of 30 endpoints fully implemented. 2 stubs remaining.**
