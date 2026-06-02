"""
Table-driven auth-gating matrix.

Every protected router must reject requests that carry no Authorization
header with a 403 (FastAPI HTTPBearer returns 403 when the header is
absent) or 401 when a token is present but invalid.
"""
import pytest

# (method, url)
PROTECTED_ENDPOINTS = [
    ("POST", "/ocr/extract"),
    ("POST", "/ocr/extract-batch"),
    ("POST", "/asag/grade"),
    ("POST", "/bkt/update"),
    ("POST", "/api/v1/agents/teacher/assessment-generation"),
    ("POST", "/api/v1/agents/student/assessment"),
    ("POST", "/content/generate"),
    ("POST", "/resources/upload"),
    ("POST", "/assessment-gen/generate"),
    ("POST", "/ai-tutor/introduce"),
    ("POST", "/ai-tutor/coaching"),
    ("POST", "/syllabus-extraction/extract"),
    ("POST", "/rag/inject"),
    ("POST", "/rag/query"),
    ("POST", "/rag/coverage-query"),
    ("GET",  "/rag/status/test-subject"),
    ("DELETE", "/rag/documents/test-doc?subject_id=test-subject"),
    ("POST", "/notes/generate"),
    ("POST", "/notes/chat"),
    ("POST", "/persisted-notes/generate"),
    ("POST", "/reteach/generate-cards"),
    ("POST", "/developmentplan/generate-missions"),
]


@pytest.mark.parametrize("method,url", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_no_token(client, method, url):
    """No Authorization header → 403 (HTTPBearer missing credentials)."""
    response = getattr(client, method.lower())(url)
    assert response.status_code == 403, (
        f"{method} {url} expected 403, got {response.status_code}: {response.text}"
    )


@pytest.mark.parametrize("method,url", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_bad_token(client, method, url):
    """Invalid JWT → 401."""
    headers = {"Authorization": "Bearer this.is.not.a.valid.jwt"}
    response = getattr(client, method.lower())(url, headers=headers)
    assert response.status_code == 401, (
        f"{method} {url} expected 401, got {response.status_code}: {response.text}"
    )
