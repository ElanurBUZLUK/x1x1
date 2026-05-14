from __future__ import annotations

from src.schemas import KPSSQuestion, OpenAnswerAttemptEvent, OpenAnswerEvaluation, UserAnswerEvent


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


def make_open_answer_attempt_event(
    *,
    user_id: str,
    question_id: str,
    lesson: str,
    topic: str,
    difficulty: float,
    student_answer: str,
    reference_answer: str,
    evaluation: OpenAnswerEvaluation,
    response_time: float | None = None,
) -> OpenAnswerAttemptEvent:
    """Açık uçlu cevap değerlendirmesini öğrenci geçmişi olayına çevirir."""
    return OpenAnswerAttemptEvent(
        user_id=user_id,
        question_id=question_id,
        lesson=lesson,
        topic=topic,
        difficulty=difficulty,
        is_correct=evaluation.is_correct,
        answer_score=evaluation.score,
        evaluation_label=evaluation.label,
        confidence=evaluation.confidence,
        student_answer=student_answer,
        reference_answer=reference_answer,
        matched_concepts=evaluation.matched_concepts,
        missing_concepts=evaluation.missing_concepts,
        feedback=evaluation.feedback,
        response_time=response_time,
    )
