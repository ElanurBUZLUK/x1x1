import pytest
from pydantic import ValidationError

from src.schemas import KPSSQuestion


def _valid_question(**overrides):
    data = {
        "question_id": "q1",
        "section": "Genel Yetenek",
        "lesson": "Türkçe",
        "topic": "Paragraf",
        "difficulty": 0.5,
        "question_text": "Ana düşünce nedir?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"},
        "correct_answer": "B",
        "explanation": "Ana düşünce açıklaması.",
    }
    data.update(overrides)
    return data


def test_valid_question_schema_accepts_backward_compatible_defaults():
    question = KPSSQuestion.model_validate(_valid_question())
    assert question.exam == "KPSS"
    assert question.source == "kpss_question_bank"
    assert question.correct_answer == "B"


def test_invalid_options_raise_validation_error():
    with pytest.raises(ValidationError):
        KPSSQuestion.model_validate(_valid_question(options={"A": "a", "B": "b"}))


def test_invalid_answer_raises_validation_error():
    with pytest.raises(ValidationError):
        KPSSQuestion.model_validate(_valid_question(correct_answer="Z"))
