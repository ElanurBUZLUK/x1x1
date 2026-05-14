from src.schemas import OpenAnswerAttemptEvent, UserAnswerEvent
from src.user_profile import build_student_profile


def test_profile_tracks_weak_topics_and_solved_ids_without_duplicates():
    history = [
        UserAnswerEvent(user_id="u1", question_id="q1", lesson="Tarih", topic="Osmanlı", difficulty=0.4, is_correct=False),
        UserAnswerEvent(user_id="u1", question_id="q1", lesson="Tarih", topic="Osmanlı", difficulty=0.4, is_correct=True),
        UserAnswerEvent(user_id="u1", question_id="q2", lesson="Türkçe", topic="Paragraf", difficulty=0.6, is_correct=True),
    ]
    profile = build_student_profile(history, user_id="u1")
    assert profile.solved_question_ids == ["q1", "q2"]
    assert profile.weak_topics[0] == "Tarih/Osmanlı"
    assert 0.0 <= profile.overall_level <= 1.0


def test_profile_uses_open_answer_score():
    history = [
        UserAnswerEvent(
            user_id="u1",
            question_id="q1",
            lesson="Tarih",
            topic="Osmanlı",
            difficulty=0.4,
            is_correct=False,
        ),
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

    assert profile.accuracy == 0.4
    assert profile.solved_question_ids == ["q1", "open1"]
    assert round(profile.topic_mastery["Tarih/Osmanlı"], 2) == 0.43
