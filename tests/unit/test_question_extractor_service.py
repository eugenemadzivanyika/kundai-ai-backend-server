"""Unit tests for question_extractor_service."""
import json
import pytest
from unittest.mock import AsyncMock


class TestExtractQuestionsFromText:
    @pytest.mark.asyncio
    async def test_returns_list_of_questions(self, monkeypatch):
        questions = [
            {"question_text": "What is 2+2?", "marks_hint": 1, "topic_hint": "arithmetic"},
            {"question_text": "Solve x+3=7", "marks_hint": 2, "topic_hint": "algebra"},
        ]
        monkeypatch.setattr(
            "services.question_extractor_service.call_llm",
            AsyncMock(return_value=json.dumps(questions)),
        )
        from services.question_extractor_service import extract_questions_from_text
        result = await extract_questions_from_text("dummy exam text")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["question_text"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_handles_wrapped_json_object(self, monkeypatch):
        wrapped = {"questions": [{"question_text": "Find x", "marks_hint": 3, "topic_hint": "algebra"}]}
        monkeypatch.setattr(
            "services.question_extractor_service.call_llm",
            AsyncMock(return_value=json.dumps(wrapped)),
        )
        from services.question_extractor_service import extract_questions_from_text
        result = await extract_questions_from_text("text")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_invalid_json(self, monkeypatch):
        monkeypatch.setattr(
            "services.question_extractor_service.call_llm",
            AsyncMock(return_value="not valid json at all"),
        )
        from services.question_extractor_service import extract_questions_from_text
        result = await extract_questions_from_text("text")
        assert result == []

    @pytest.mark.asyncio
    async def test_caps_at_max_questions(self, monkeypatch):
        big_list = [{"question_text": f"Q{i}", "marks_hint": 1, "topic_hint": None} for i in range(200)]
        monkeypatch.setattr(
            "services.question_extractor_service.call_llm",
            AsyncMock(return_value=json.dumps(big_list)),
        )
        from services.question_extractor_service import extract_questions_from_text, MAX_QUESTIONS
        result = await extract_questions_from_text("big paper")
        assert len(result) == MAX_QUESTIONS

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_llm_exception(self, monkeypatch):
        monkeypatch.setattr(
            "services.question_extractor_service.call_llm",
            AsyncMock(side_effect=RuntimeError("LLM unreachable")),
        )
        from services.question_extractor_service import extract_questions_from_text
        result = await extract_questions_from_text("text")
        assert result == []
