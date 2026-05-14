from src.open_answer_evaluator import OpenAnswerEvaluator


class FakeTransformerScorer:
    def predict_score_0_10(self, *, question: str, reference_answer: str, student_answer: str) -> float:
        return 8.0


def test_exact_open_answer_is_correct():
    evaluator = OpenAnswerEvaluator(use_llm=False)

    result = evaluator.evaluate(
        question_text="Türkiye Cumhuriyeti'nin başkenti neresidir?",
        reference_answer="Ankara",
        student_answer="Ankara'dır",
        accepted_aliases=["Ankara", "Ankaradır"],
        key_concepts=["Ankara"],
    )

    assert result.is_correct is True
    assert result.label == "correct"
    assert result.score >= 0.65


def test_alias_open_answer_is_correct():
    evaluator = OpenAnswerEvaluator(use_llm=False)

    result = evaluator.evaluate(
        question_text="Kanunları yayımlama yetkisi kime aittir?",
        reference_answer="Cumhurbaşkanı",
        student_answer="cumhur başkanı",
        accepted_aliases=[
            "cumhurbaşkanı",
            "cumhur başkanı",
            "Türkiye Cumhurbaşkanı",
        ],
        key_concepts=["Cumhurbaşkanı"],
    )

    assert result.is_correct is True
    assert result.label == "correct"


def test_wrong_open_answer_is_incorrect():
    evaluator = OpenAnswerEvaluator(use_llm=False)

    result = evaluator.evaluate(
        question_text="Kanunları yayımlama yetkisi kime aittir?",
        reference_answer="Cumhurbaşkanı",
        student_answer="Türkiye Büyük Millet Meclisi",
        accepted_aliases=["cumhurbaşkanı", "cumhur başkanı"],
        key_concepts=["Cumhurbaşkanı"],
    )

    assert result.is_correct is False
    assert result.label in {"incorrect", "partially_correct"}
    assert "Cumhurbaşkanı" in result.expected_answer


def test_number_word_alias_is_correct():
    evaluator = OpenAnswerEvaluator(use_llm=False)

    result = evaluator.evaluate(
        question_text="Kanun-i Esasi hangi yılda ilan edildi?",
        reference_answer="1876",
        student_answer="bin sekiz yüz yetmiş altı",
        accepted_aliases=[
            "1876",
            "bin sekiz yüz yetmiş altı",
        ],
        key_concepts=["1876"],
    )

    assert result.is_correct is True


def test_wrong_if_mentions_forces_incorrect():
    evaluator = OpenAnswerEvaluator(use_llm=False)

    result = evaluator.evaluate(
        question_text="Kanunları yayımlama yetkisi kime aittir?",
        reference_answer="Cumhurbaşkanı",
        student_answer="Bence Türkiye Büyük Millet Meclisi yayımlar.",
        accepted_aliases=["cumhurbaşkanı", "cumhur başkanı"],
        key_concepts=["Cumhurbaşkanı"],
        wrong_if_mentions=["Türkiye Büyük Millet Meclisi", "TBMM"],
    )

    assert result.is_correct is False
    assert result.label == "incorrect"
    assert result.score == 0.05
    assert result.confidence == 0.95
    assert result.misconception is not None
    assert "Türkiye Büyük Millet Meclisi" in result.misconception


def test_transformer_score_is_used_when_available_for_non_exact_answer():
    evaluator = OpenAnswerEvaluator(use_llm=False)
    evaluator.transformer_scorer = FakeTransformerScorer()

    result = evaluator.evaluate(
        question_text="Fotosentez nedir?",
        reference_answer=(
            "Fotosentez, bitkilerin ışık enerjisini kullanarak karbondioksit "
            "ve sudan glikoz üretmesi ve oksijen açığa çıkarmasıdır."
        ),
        student_answer="Bitkiler güneş ışığıyla kendi besinini üretir.",
        key_concepts=["ışık enerjisi", "besin üretimi"],
    )

    assert result.score == 0.8
    assert result.label == "correct"
    assert result.grading_basis == "transformer_regression_reference_answer_and_context"
    assert result.used_llm is False
