"""Unit tests for bkt_service — pure math, no mocking needed."""
import pytest
from services.bkt_service import update_mastery


class TestUpdateMastery:
    def test_correct_answer_increases_mastery(self):
        result = update_mastery({"prev_mastery": 0.5, "correct": True})
        assert result["new_mastery"] > 0.5

    def test_incorrect_answer_with_high_mastery(self):
        result = update_mastery({"prev_mastery": 0.9, "correct": False})
        assert "new_mastery" in result
        assert 0.0 < result["new_mastery"] <= 1.0

    def test_mastery_capped_at_09999(self):
        result = update_mastery({"prev_mastery": 0.99, "correct": True, "p_transit": 0.99})
        assert result["new_mastery"] <= 0.9999

    def test_response_shape(self):
        result = update_mastery({"prev_mastery": 0.3, "correct": True, "attribute_id": "MATH-001"})
        assert set(result.keys()) == {"new_mastery", "attribute_id", "growth", "timestamp"}

    def test_attribute_id_passthrough(self):
        result = update_mastery({"prev_mastery": 0.4, "correct": False, "attribute_id": "MATH-XYZ"})
        assert result["attribute_id"] == "MATH-XYZ"

    def test_growth_is_difference(self):
        prev = 0.4
        result = update_mastery({"prev_mastery": prev, "correct": True})
        expected_growth = round(result["new_mastery"] - prev, 4)
        assert result["growth"] == expected_growth

    def test_default_parameters_used_when_missing(self):
        # Calling with empty payload should not raise; defaults should apply
        result = update_mastery({})
        assert 0.0 < result["new_mastery"] <= 1.0

    @pytest.mark.parametrize("prev,correct", [
        (0.1, True),
        (0.1, False),
        (0.5, True),
        (0.5, False),
        (0.9, True),
        (0.9, False),
    ])
    def test_result_always_in_valid_range(self, prev, correct):
        result = update_mastery({"prev_mastery": prev, "correct": correct})
        assert 0.0 < result["new_mastery"] <= 1.0
