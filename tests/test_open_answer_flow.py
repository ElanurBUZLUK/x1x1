from main_api import SubmitOpenAnswerRequest, submit_open_answer


def test_submit_open_answer_returns_evaluation_and_updated_profile_without_persisting():
    request = SubmitOpenAnswerRequest(
        user_id="u_open",
        question_id="open_democracy_1",
        question_text="Demokrasi nedir?",
        reference_answer="Demokrasi, halkın yönetime doğrudan veya temsilcileri aracılığıyla katıldığı yönetim biçimidir.",
        student_answer="Halkın seçim yoluyla yönetime katıldığı sistemdir.",
        lesson="Vatandaşlık",
        topic="Demokrasi",
        difficulty=0.5,
        key_concepts=["halk", "yönetim", "seçim"],
        history=[],
        persist=False,
        use_llm=False,
    )

    response = submit_open_answer(request)

    assert response.evaluation.score >= 0.0
    assert response.attempt_event.answer_type == "open_answer"
    assert response.attempt_event.answer_score == response.evaluation.score
    assert response.updated_profile.solved_question_ids == ["open_democracy_1"]
    assert response.updated_user_level_preview == response.updated_profile.overall_level
