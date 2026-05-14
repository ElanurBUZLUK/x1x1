from src.schemas import KPSSQuestion


def _valid_question(**overrides):
    data = {
        "question_id": "q1",
        "section": "Genel Yetenek",
        "lesson": "Türkçe",
        "topic": "Paragraf",
        "difficulty": 0.5,
        "question_text": "Ana düşünce nedir?",
        "reference_answer": "Ana düşünce, paragrafta asıl anlatılmak istenen temel fikirdir.",
        "accepted_aliases": ["Ana fikir"],
        "key_concepts": ["ana düşünce", "temel fikir"],
        "explanation": "Ana düşünce açıklaması.",
    }
    data.update(overrides)
    return data


def test_valid_question_schema_accepts_backward_compatible_defaults():
    question = KPSSQuestion.model_validate(_valid_question())
    assert question.exam == "KPSS"
    assert question.source == "kpss_question_bank"
    assert question.reference_answer.startswith("Ana düşünce")


def test_embedding_text_contains_open_answer_fields():
    question = KPSSQuestion.model_validate(_valid_question())
    text = question.embedding_text()
    assert "Referans cevap:" in text
    assert "Ana kavramlar:" in text
