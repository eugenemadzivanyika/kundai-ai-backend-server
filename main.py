import os
from dotenv import load_dotenv
load_dotenv()  # Must run before any module that reads os.getenv() at import time

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import bkt, developmentPlan, devplan_content_generation, health, ocr, asag, agents, content, resources, assessment_generation, ai_tutor, syllabus_extraction, rag, notes, persisted_notes, reteach
from middleware.error_middleware import ErrorLoggingMiddleware
from middleware.auth import require_user

# List the origins you trust
origins = [
    "http://localhost:5173",  # Your React Frontend
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="KundAI AI Services Backend",
    version="0.1.0",
    description="FastAPI service for AI/ML endpoints.",
)

app.add_middleware(ErrorLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allow Content-Type, Authorization, etc.
)

_auth = [Depends(require_user)]

app.include_router(health.router)  # public — polled by the Node health check
app.include_router(ocr.router, dependencies=_auth)
app.include_router(asag.router, dependencies=_auth)
app.include_router(bkt.router, dependencies=_auth)
app.include_router(agents.router, dependencies=_auth)
app.include_router(developmentPlan.router, dependencies=_auth)
app.include_router(content.router, dependencies=_auth)
app.include_router(resources.router, dependencies=_auth)
app.include_router(devplan_content_generation.router, dependencies=_auth)
app.include_router(assessment_generation.router, dependencies=_auth)
app.include_router(ai_tutor.router, dependencies=_auth)
app.include_router(syllabus_extraction.router, dependencies=_auth)
app.include_router(rag.router, dependencies=_auth)
app.include_router(notes.router, dependencies=_auth)
app.include_router(persisted_notes.router, dependencies=_auth)
app.include_router(reteach.router, dependencies=_auth)
# /uploads is NOT mounted as public static — serve files through an authenticated endpoint instead
