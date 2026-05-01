from __future__ import annotations

from src.schemas import KPSSQuestion, UserAnswerEvent


def make_answer_event(
    user_id: str,
    question: KPSSQuestion,
    user_answer: str,
    response_time: float | None = None,
) -> UserAnswerEvent:
    """Yeni cevap olayını normalize eder. Kalıcı veritabanı backend tarafında tutulabilir."""
    return UserAnswerEvent(
        user_id=user_id,
        question_id=question.question_id,
        lesson=question.lesson,
        topic=question.topic,
        difficulty=question.difficulty,
        is_correct=user_answer.strip().upper() == question.correct_answer,
        response_time=response_time,
    )
