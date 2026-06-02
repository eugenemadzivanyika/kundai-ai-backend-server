"""
Shared fixtures for Phase 2 tests.

JWT tokens are signed with HS256 + a test secret so no real key material
is needed.  LLM calls (call_llm) and ChromaDB are patched at the module
level via autouse fixtures so no network or paid API keys are ever hit.
"""
import os
import time
import pytest
import jwt
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Set test env vars BEFORE importing the app so auth.py picks them up.
TEST_JWT_SECRET = "test-secret-for-kundai-phase2"
os.environ.setdefault("JWT_SECRET", TEST_JWT_SECRET)
os.environ.setdefault("NODE_ENV", "test")
# Prevent real LLM / storage calls from needing real keys
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
# Empty string is intentional: falsy so the OCR router skips the Gemini path,
# but "GEMINI_API_KEY" in os.environ is still True — add a comment if code ever
# switches to an `in` check so this default is updated accordingly.
os.environ.setdefault("GEMINI_API_KEY", "")

from main import app  # noqa: E402  (must come after env setup)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Synchronous TestClient — works for all routes including async ones."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def make_token(role: str = "student", user_id: str = "000000000000000000000001") -> str:
    """Sign a token the same way the Node backend does (HS256 fallback)."""
    secret = os.environ["JWT_SECRET"]
    payload = {
        "id": user_id,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture(scope="session")
def auth_header():
    """Bearer header for a regular student."""
    return {"Authorization": f"Bearer {make_token('student')}"}


@pytest.fixture(scope="session")
def teacher_header():
    """Bearer header for a teacher."""
    return {"Authorization": f"Bearer {make_token('teacher')}"}


# ---------------------------------------------------------------------------
# LLM mock — autouse so no test can accidentally hit Mistral/Gemini
# ---------------------------------------------------------------------------

LLM_DEFAULT_RESPONSE = '{"result": "mocked"}'


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Patch services.llm_service.call_llm globally."""
    async_mock = AsyncMock(return_value=LLM_DEFAULT_RESPONSE)
    monkeypatch.setattr("services.llm_service.call_llm", async_mock)
    # Modules that imported call_llm directly at import time must also be patched
    # here. If you add a new service that does `from services.llm_service import
    # call_llm` at module level, add it to this list — otherwise the LLM will be
    # called for real in tests with no warning.
    for mod in [
        "services.asag_service",
        "services.assessment_generation_service",
        "services.content_service",
        "services.agents_service",
        "services.ai_tutor_service",
        "services.notes_service",
        "services.reteach_service",
        "services.developmentPlan_service",
        "services.syllabus_extraction_service",
        "services.document_classifier_service",
        "services.question_extractor_service",
    ]:
        try:
            monkeypatch.setattr(f"{mod}.call_llm", async_mock)
        except AttributeError:
            pass
    return async_mock


# ---------------------------------------------------------------------------
# ChromaDB mock — autouse so no chroma client is created
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_chromadb(monkeypatch):
    """Replace chromadb.Client / PersistentClient with a MagicMock."""
    mock_collection = MagicMock()
    mock_collection.add = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = {
        "documents": [["mock doc"]],
        "metadatas": [[{"source": "test"}]],
        "distances": [[0.1]],
    }
    mock_collection.get.return_value = {"ids": []}

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.get_collection.return_value = mock_collection

    monkeypatch.setattr("chromadb.PersistentClient", MagicMock(return_value=mock_client))
    monkeypatch.setattr("chromadb.Client", MagicMock(return_value=mock_client))
    return mock_client


# ---------------------------------------------------------------------------
# RAG service mock — autouse so no Mistral embedding calls are made
# ---------------------------------------------------------------------------

RAG_EMPTY_RESULT = {
    "chunks": [],
    "sources": [],
    "distances": [],
    "has_documents": False,
}


@pytest.fixture(autouse=True)
def mock_rag_service(monkeypatch):
    """Patch rag_service query/embed functions so no network calls are made."""
    async_query = AsyncMock(return_value=RAG_EMPTY_RESULT)
    async_coverage = AsyncMock(return_value={"coverage": []})
    async_embed = AsyncMock(return_value=[[0.1] * 4])
    async_inject = AsyncMock(return_value=1)

    monkeypatch.setattr("services.rag_service.query_rag", async_query)
    monkeypatch.setattr("services.rag_service.embed_texts", async_embed)
    monkeypatch.setattr("services.rag_service.inject_document", async_inject)
    monkeypatch.setattr("services.rag_service.compute_attribute_coverage", async_coverage)

    # Also patch in modules that imported query_rag at import time
    for mod in ("services.notes_service", "services.ai_tutor_service"):
        try:
            monkeypatch.setattr(f"{mod}.rag_service.query_rag", async_query)
        except AttributeError:
            pass

    return async_query
