from src.schemas import UserAnswerEvent
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
