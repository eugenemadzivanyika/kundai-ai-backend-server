"""Unit tests for document_classifier_service — heuristic and LLM paths."""
import json
import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Heuristic path — no LLM call needed
# ---------------------------------------------------------------------------

class TestHeuristicClassify:
    def test_marks_pattern_classifies_as_question_paper(self):
        from services.document_classifier_service import _heuristic_classify
        text = "Q1. Calculate [2] Q2. Find [3] Q3. Solve [1] Q4. Express [4] Q5. Evaluate [2] Q6. Show [3] Q7. Prove [2] Q8. Expand [1]"
        assert _heuristic_classify(text) == "question_paper"

    def test_numbered_questions_classifies_as_question_paper(self):
        from services.document_classifier_service import _heuristic_classify
        lines = "\n".join([
            f"{i}. Calculate the value of x" for i in range(1, 10)
        ])
        result = _heuristic_classify(lines)
        assert result == "question_paper"

    def test_prose_text_returns_none(self):
        from services.document_classifier_service import _heuristic_classify
        text = "Quadratic equations are polynomials of degree two. They appear in many real-world situations."
        assert _heuristic_classify(text) is None


# ---------------------------------------------------------------------------
# Build sample helper
# ---------------------------------------------------------------------------

class TestBuildSample:
    def test_short_text_returned_unchanged(self):
        from services.document_classifier_service import _build_sample
        short = "word " * 100
        assert _build_sample(short) == short

    def test_long_text_sampled_from_three_regions(self):
        from services.document_classifier_service import _build_sample
        long_text = "word " * 2000
        result = _build_sample(long_text)
        assert "[...]" in result


# ---------------------------------------------------------------------------
# Full classify_document — LLM path mocked
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    @pytest.mark.asyncio
    async def test_classifies_question_paper_via_heuristic(self):
        from services.document_classifier_service import classify_document
        text = " ".join(["[2] " * 5 + "Evaluate " for _ in range(3)])
        result = await classify_document(text)
        assert result == "question_paper"

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_for_ambiguous_text(self, monkeypatch):
        monkeypatch.setattr(
            "services.document_classifier_service.call_llm",
            AsyncMock(return_value=json.dumps({"type": "learning_material"})),
        )
        from services.document_classifier_service import classify_document
        result = await classify_document("This is a textbook chapter about algebra.")
        assert result == "learning_material"

    @pytest.mark.asyncio
    async def test_llm_returns_question_paper(self, monkeypatch):
        monkeypatch.setattr(
            "services.document_classifier_service.call_llm",
            AsyncMock(return_value=json.dumps({"type": "question_paper"})),
        )
        from services.document_classifier_service import classify_document
        result = await classify_document("Some ambiguous document text here.")
        assert result == "question_paper"

    @pytest.mark.asyncio
    async def test_defaults_to_learning_material_on_llm_error(self, monkeypatch):
        monkeypatch.setattr(
            "services.document_classifier_service.call_llm",
            AsyncMock(side_effect=RuntimeError("LLM down")),
        )
        from services.document_classifier_service import classify_document
        result = await classify_document("Some text without strong signals.")
        assert result == "learning_material"

    @pytest.mark.asyncio
    async def test_defaults_to_learning_material_on_invalid_llm_json(self, monkeypatch):
        monkeypatch.setattr(
            "services.document_classifier_service.call_llm",
            AsyncMock(return_value="not json"),
        )
        from services.document_classifier_service import classify_document
        result = await classify_document("Some ambiguous text here.")
        assert result == "learning_material"
