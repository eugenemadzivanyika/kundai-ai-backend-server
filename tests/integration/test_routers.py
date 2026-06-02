"""
Happy-path contract tests — one per router.

All external calls (LLM, ChromaDB, RAG embeddings) are suppressed by the
autouse fixtures in conftest.py.  Tests here mock at the service boundary
for anything that requires real files or additional network calls.
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# BKT — pure math, no external calls
# ---------------------------------------------------------------------------

class TestBkt:
    def test_update_returns_new_mastery(self, client, auth_header):
        payload = {
            "prev_mastery": 0.5,
            "correct": True,
            "p_guess": 0.2,
            "p_slip": 0.1,
            "p_transit": 0.1,
            "attribute_id": "MATH-F1-RN-01",
        }
        resp = client.post("/bkt/update", json=payload, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "new_mastery" in data
        assert 0 < data["new_mastery"] <= 1.0
        assert data["attribute_id"] == "MATH-F1-RN-01"

    def test_update_correct_increases_mastery(self, client, auth_header):
        base = {"prev_mastery": 0.3, "correct": True}
        resp = client.post("/bkt/update", json=base, headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["new_mastery"] > 0.3

    def test_update_incorrect_response_shape(self, client, auth_header):
        base = {"prev_mastery": 0.8, "correct": False}
        resp = client.post("/bkt/update", json=base, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "growth" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# ASAG — mocked call_llm via autouse
# ---------------------------------------------------------------------------

class TestAsag:
    def test_grade_happy_path(self, client, auth_header):
        payload = {
            "questions": [
                {
                    "number": 1,
                    "text": "What is 2+2?",
                    "correctAnswer": "4",
                    "studentAnswer": "4",
                    "maxPoints": 2,
                }
            ],
            "studentContext": {"currentMastery": 0.5},
        }
        resp = client.post("/asag/grade", json=payload, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "grading" in data
        assert "profile" in data


# ---------------------------------------------------------------------------
# Content — stub, no mocking needed
# ---------------------------------------------------------------------------

class TestContent:
    def test_generate_returns_200(self, client, auth_header):
        resp = client.post("/content/generate", json={"topic": "Algebra"}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "content"


# ---------------------------------------------------------------------------
# Assessment generation
# ---------------------------------------------------------------------------

class TestAssessmentGeneration:
    def test_generate_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"questions": [{"id": 1, "stem": "Test?"}]}
        monkeypatch.setattr(
            "services.assessment_generation_service.generate_syllabus_questions",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "syllabus_context": [{"name": "Algebra", "attribute_id": "MATH-F1-ALG-01"}],
            "count": 5,
            "difficulty": "medium",
        }
        resp = client.post("/assessment-gen/generate", json=payload, headers=auth_header)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Development plan
# ---------------------------------------------------------------------------

class TestDevelopmentPlan:
    def test_generate_missions_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"missions": [{"task": "Practice algebra"}]}
        monkeypatch.setattr(
            "services.developmentPlan_service.generate_surgical_missions",
            AsyncMock(return_value=mock_result),
        )
        payload = {"attribute_id": "MATH-F1-RN-02", "initial_mastery": 0.12}
        resp = client.post("/developmentplan/generate-missions", json=payload, headers=auth_header)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class TestNotes:
    def test_generate_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"notes": "## Topic\nSome content", "sources": [], "grounded_by_rag": False}
        monkeypatch.setattr(
            "services.notes_service.generate_notes",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "subject_id": "subj-001",
            "topic": "Quadratic Equations",
            "attribute_name": "MATH-F2-ALG-03",
            "level": "O Level",
        }
        resp = client.post("/notes/generate", json=payload, headers=auth_header)
        assert resp.status_code == 200
        assert "notes" in resp.json()

    def test_chat_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"answer": "Great question!", "grounded_by_rag": False}
        monkeypatch.setattr(
            "services.notes_service.answer_topic_question",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "subject_id": "subj-001",
            "topic": "Quadratic Equations",
            "notes_context": "## Overview\nQuadratics are ...",
            "question": "What is a quadratic?",
            "history": [],
        }
        resp = client.post("/notes/chat", json=payload, headers=auth_header)
        assert resp.status_code == 200
        assert "answer" in resp.json()


# ---------------------------------------------------------------------------
# Persisted notes
# ---------------------------------------------------------------------------

class TestPersistedNotes:
    def test_generate_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"notes": "## Topic\nContent", "sources": [], "grounded_by_rag": False}
        monkeypatch.setattr(
            "services.notes_service.generate_notes",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "subject_id": "subj-001",
            "topic_name": "Algebra",
            "subtopic_name": "Quadratic Equations",
            "level": "O Level",
        }
        resp = client.post("/persisted-notes/generate", json=payload, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert "persisted" in data


# ---------------------------------------------------------------------------
# Reteach
# ---------------------------------------------------------------------------

class TestReteach:
    def test_generate_cards_empty_topics(self, client, auth_header):
        payload = {"class_id": "cls-001", "subject": "Mathematics", "topics": []}
        resp = client.post("/reteach/generate-cards", json=payload, headers=auth_header)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_generate_cards_happy_path(self, client, auth_header, monkeypatch):
        mock_card = {
            "topic": "Algebra",
            "subtopic": "Quadratics",
            "avg_mastery": 0.3,
            "difficulty_score": 0.7,
            "intervention_script": "Hello class...",
            "exit_ticket": [],
        }
        monkeypatch.setattr(
            "services.reteach_service.generate_reteach_card",
            AsyncMock(return_value=mock_card),
        )
        payload = {
            "class_id": "cls-001",
            "subject": "Mathematics",
            "topics": [{"topic": "Algebra", "subtopic": "Quadratics", "avg_mastery": 0.3}],
        }
        resp = client.post("/reteach/generate-cards", json=payload, headers=auth_header)
        assert resp.status_code == 200
        cards = resp.json()
        assert len(cards) == 1
        assert cards[0]["topic"] == "Algebra"


# ---------------------------------------------------------------------------
# AI Tutor
# ---------------------------------------------------------------------------

class TestAiTutor:
    def test_introduce_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"introduction": "Welcome to the lesson!"}
        monkeypatch.setattr(
            "routers.ai_tutor.ai_tutor_service.generate_plan_introduction",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "profile": {"strengths": [], "deficiencies": []},
            "plan": {"missions": []},
        }
        resp = client.post("/ai-tutor/introduce", json=payload, headers=auth_header)
        assert resp.status_code == 200
        assert "introduction" in resp.json()

    def test_coaching_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"guidance": "Good thinking!", "checkpointPassed": False}
        monkeypatch.setattr(
            "routers.ai_tutor.ai_tutor_service.generate_coaching_response",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "message": "I think it is x = 2",
            "history": [],
            "context": {"mode": "socratic"},
        }
        resp = client.post("/ai-tutor/coaching", json=payload, headers=auth_header)
        assert resp.status_code == 200
        assert "guidance" in resp.json()


# ---------------------------------------------------------------------------
# Syllabus extraction
# ---------------------------------------------------------------------------

class TestSyllabusExtraction:
    def test_extract_happy_path(self, client, auth_header, monkeypatch):
        import base64
        mock_result = {"attributes": [{"attribute_id": "MATH-F1-RN-01", "name": "Number systems"}]}
        # Router does `from services.syllabus_extraction_service import extract_attributes_from_syllabus`
        # so we must patch the name in the router's own namespace.
        monkeypatch.setattr(
            "routers.syllabus_extraction.extract_attributes_from_syllabus",
            AsyncMock(return_value=mock_result),
        )
        payload = {
            "file_content_b64": base64.b64encode(b"dummy syllabus content").decode(),
            "mime_type": "application/pdf",
        }
        resp = client.post("/syllabus-extraction/extract", json=payload, headers=auth_header)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class TestAgents:
    def test_teacher_assessment_generation_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"questions": [{"id": 1, "stem": "Solve 2x=4"}]}
        # Router imports `generate_assessment_from_resource` directly, patch in router namespace.
        monkeypatch.setattr(
            "routers.agents.generate_assessment_from_resource",
            AsyncMock(return_value=mock_result),
        )
        resp = client.post(
            "/api/v1/agents/teacher/assessment-generation",
            json={"resource_id": "some-file-id.pdf"},
            headers=auth_header,
        )
        assert resp.status_code == 200

    def test_teacher_assessment_missing_resource_id(self, client, auth_header):
        resp = client.post(
            "/api/v1/agents/teacher/assessment-generation",
            json={},
            headers=auth_header,
        )
        assert resp.status_code == 400

    def test_student_assessment_text_happy_path(self, client, auth_header, monkeypatch):
        mock_assessment = {
            "marks_achieved": 7,
            "total_possible_marks": 10,
            "overall_feedback": "Good work",
        }
        # Router imports call_llm and parse_json_from_ai directly; patch both in its namespace.
        monkeypatch.setattr("routers.agents.call_llm", AsyncMock(return_value='{"result":"ok"}'))
        monkeypatch.setattr("routers.agents.parse_json_from_ai", MagicMock(return_value=mock_assessment))
        resp = client.post(
            "/api/v1/agents/student/assessment",
            data={"text": "The answer is x = 2", "module": "Mathematics"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "assessment" in data
        assert data["content_type"] == "text"

    def test_student_assessment_no_input_returns_400(self, client, auth_header):
        resp = client.post(
            "/api/v1/agents/student/assessment",
            data={},
            headers=auth_header,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class TestResources:
    def test_upload_happy_path(self, client, auth_header, monkeypatch):
        monkeypatch.setattr(
            "services.storage_service.save_upload",
            AsyncMock(return_value="uploads/test-file-id.pdf"),
        )
        file_bytes = b"%PDF-1.4 fake pdf"
        resp = client.post(
            "/resources/upload",
            files={"file": ("test.pdf", io.BytesIO(file_bytes), "application/pdf")},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "url" in data

    def test_upload_invalid_type_returns_400(self, client, auth_header):
        resp = client.post(
            "/resources/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            headers=auth_header,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

class TestOcr:
    def test_extract_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"pages": [{"raw_markdown": "Extracted text"}]}
        # GEMINI_API_KEY is already "" so router goes to perform_ocr.
        # Router imports perform_ocr directly; patch its namespace binding.
        monkeypatch.setattr("routers.ocr.perform_ocr", AsyncMock(return_value=mock_result))

        file_bytes = b"\x89PNG\r\nfake image"
        resp = client.post(
            "/ocr/extract",
            files={"file": ("test.png", io.BytesIO(file_bytes), "image/png")},
            headers=auth_header,
        )
        assert resp.status_code == 200

    def test_extract_unsupported_type_returns_415(self, client, auth_header):
        resp = client.post(
            "/ocr/extract",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            headers=auth_header,
        )
        assert resp.status_code == 415

    def test_extract_batch_happy_path(self, client, auth_header, monkeypatch):
        mock_result = {"pages": [{"raw_markdown": "Extracted text"}]}
        monkeypatch.setattr("routers.ocr.perform_ocr", AsyncMock(return_value=mock_result))

        file_bytes = b"\x89PNG\r\nfake image"
        resp = client.post(
            "/ocr/extract-batch",
            files=[("files", ("test.png", io.BytesIO(file_bytes), "image/png"))],
            headers=auth_header,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

class TestRag:
    def test_query_happy_path(self, client, auth_header):
        payload = {"subject_id": "subj-001", "query": "What is algebra?", "n_results": 3}
        resp = client.post("/rag/query", json=payload, headers=auth_header)
        assert resp.status_code == 200

    def test_coverage_query_happy_path(self, client, auth_header):
        payload = {
            "subject_id": "subj-001",
            "attributes": [{"id": "MATH-F1-RN-01", "name": "Numbers", "description": "Real numbers"}],
        }
        resp = client.post("/rag/coverage-query", json=payload, headers=auth_header)
        assert resp.status_code == 200

    def test_status_happy_path(self, client, auth_header):
        resp = client.get("/rag/status/subj-001", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "chunk_count" in data

    def test_delete_document_happy_path(self, client, auth_header):
        resp = client.delete(
            "/rag/documents/test-doc-id?subject_id=subj-001",
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert "deleted" in resp.json()

    def test_inject_non_pdf_returns_400(self, client, auth_header):
        resp = client.post(
            "/rag/inject",
            data={"subject_id": "subj-001"},
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            headers=auth_header,
        )
        assert resp.status_code == 400

    def test_inject_pdf_happy_path(self, client, auth_header, monkeypatch):
        monkeypatch.setattr(
            "services.pdf_parser_service.extract_text_from_pdf",
            AsyncMock(return_value="Some textbook content about algebra."),
        )
        monkeypatch.setattr(
            "services.document_classifier_service.classify_document",
            AsyncMock(return_value="learning_material"),
        )
        monkeypatch.setattr(
            "services.rag_service.chunk_text",
            MagicMock(return_value=["chunk1", "chunk2"]),
        )

        file_bytes = b"%PDF-1.4 fake content"
        resp = client.post(
            "/rag/inject",
            data={"subject_id": "subj-001"},
            files={"file": ("notes.pdf", io.BytesIO(file_bytes), "application/pdf")},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == "subj-001"
        assert data["document_type"] == "learning_material"
