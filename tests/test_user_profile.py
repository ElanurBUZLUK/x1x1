from src.schemas import OpenAnswerAttemptEvent
from src.user_profile import build_student_profile


def test_profile_tracks_weak_topics_and_solved_ids_without_duplicates():
    history = [
        OpenAnswerAttemptEvent(
            user_id="u1",
            question_id="q1",
            lesson="Tarih",
            topic="Osmanlı",
            difficulty=0.4,
            is_correct=False,
            answer_score=0.2,
            evaluation_label="incorrect",
            confidence=0.8,
            student_answer="Yanlış cevap",
            reference_answer="Doğru cevap",
            feedback="Yanlış.",
        ),
        OpenAnswerAttemptEvent(
            user_id="u1",
            question_id="q1",
            lesson="Tarih",
            topic="Osmanlı",
            difficulty=0.4,
            is_correct=True,
            answer_score=0.8,
            evaluation_label="correct",
            confidence=0.8,
            student_answer="Doğru cevap",
            reference_answer="Doğru cevap",
            feedback="Doğru.",
        ),
        OpenAnswerAttemptEvent(
            user_id="u1",
            question_id="q2",
            lesson="Türkçe",
            topic="Paragraf",
            difficulty=0.6,
            is_correct=True,
            answer_score=0.9,
            evaluation_label="correct",
            confidence=0.8,
            student_answer="Ana düşünce",
            reference_answer="Ana düşünce",
            feedback="Doğru.",
        ),
    ]
    profile = build_student_profile(history, user_id="u1")
    assert profile.solved_question_ids == ["q1", "q2"]
    assert profile.weak_topics[0] == "Tarih/Osmanlı"
    assert 0.0 <= profile.overall_level <= 1.0


def test_profile_uses_open_answer_score():
    history = [
        OpenAnswerAttemptEvent(
            user_id="u1",
            question_id="open1",
            lesson="Tarih",
            topic="Osmanlı",
            difficulty=0.6,
            is_correct=True,
            answer_score=0.8,
            evaluation_label="correct",
            confidence=0.85,
            student_answer="Halkın seçimle yönetime katılmasıdır.",
            reference_answer="Demokrasi halkın yönetime katıldığı sistemdir.",
            feedback="Doğru.",
        ),
    ]

    profile = build_student_profile(history, user_id="u1")

    assert profile.accuracy == 0.8
    assert profile.solved_question_ids == ["open1"]
    assert round(profile.topic_mastery["Tarih/Osmanlı"], 2) == 0.74
