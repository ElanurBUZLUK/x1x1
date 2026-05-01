from src.ranking import RankingWeights, calculate_final_score


def test_final_score_formula_is_weighted_and_normalized():
    score = calculate_final_score(
        topic_match=1.0,
        difficulty_match=0.8,
        weakness_match=1.0,
        semantic_similarity=0.5,
        novelty_score=1.0,
        weights=RankingWeights(topic=0.30, difficulty=0.25, weakness=0.20, semantic=0.15, novelty=0.10),
    )
    assert round(score, 3) == 0.875


def test_final_score_is_clamped():
    score = calculate_final_score(
        topic_match=3.0,
        difficulty_match=3.0,
        weakness_match=3.0,
        semantic_similarity=3.0,
        novelty_score=3.0,
    )
    assert score == 1.0
