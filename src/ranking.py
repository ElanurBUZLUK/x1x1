from __future__ import annotations

from dataclasses import dataclass

from src.config import config
from src.schemas import CandidateQuestion, KPSSQuestion, StudentProfile


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class RankingWeights:
    topic: float = config.topic_weight
    difficulty: float = config.difficulty_weight
    weakness: float = config.weakness_weight
    semantic: float = config.semantic_weight
    novelty: float = config.novelty_weight

    def total(self) -> float:
        return self.topic + self.difficulty + self.weakness + self.semantic + self.novelty


def _same_lesson_topic_score(question: KPSSQuestion, target_topic: str | None) -> float:
    if not target_topic:
        return 0.5
    lesson, _, topic = target_topic.partition("/")
    if question.lesson == lesson and question.topic == topic:
        return 1.0
    if question.lesson == lesson:
        return 0.65
    return 0.15


def _difficulty_match(user_level: float, question_difficulty: float) -> float:
    return _clamp01(1.0 - abs(user_level - question_difficulty))


def _weakness_match(question: KPSSQuestion, target_topic: str | None) -> float:
    if not target_topic:
        return 0.4
    return 1.0 if question.topic_key() == target_topic else 0.25


def _novelty_score(question: KPSSQuestion, profile: StudentProfile) -> float:
    return 0.0 if question.question_id in set(profile.solved_question_ids) else 1.0


def calculate_final_score(
    *,
    topic_match: float,
    difficulty_match: float,
    weakness_match: float,
    semantic_similarity: float,
    novelty_score: float,
    weights: RankingWeights | None = None,
) -> float:
    """Ayrı test edilebilir final skor formülü.

    Varsayılan formül:
        0.30 topic + 0.25 difficulty + 0.20 weakness + 0.15 semantic + 0.10 novelty

    Ağırlık toplamı 1 değilse skor otomatik normalize edilir. Böylece .env ile
    ağırlık değiştirildiğinde final_score 0-1 aralığında kalır.
    """
    weights = weights or RankingWeights()
    total = weights.total() or 1.0
    raw = (
        weights.topic * _clamp01(topic_match)
        + weights.difficulty * _clamp01(difficulty_match)
        + weights.weakness * _clamp01(weakness_match)
        + weights.semantic * _clamp01(semantic_similarity)
        + weights.novelty * _clamp01(novelty_score)
    ) / total
    return _clamp01(raw)


def score_candidate(
    question: KPSSQuestion,
    profile: StudentProfile,
    target_topic: str | None,
    semantic_similarity: float,
) -> CandidateQuestion:
    topic_match = _same_lesson_topic_score(question, target_topic)
    difficulty_match = _difficulty_match(profile.overall_level, question.difficulty)
    weakness_match = _weakness_match(question, target_topic)
    novelty_score = _novelty_score(question, profile)
    final_score = calculate_final_score(
        topic_match=topic_match,
        difficulty_match=difficulty_match,
        weakness_match=weakness_match,
        semantic_similarity=semantic_similarity,
        novelty_score=novelty_score,
    )

    return CandidateQuestion(
        question=question,
        semantic_similarity=_clamp01(semantic_similarity),
        topic_match=topic_match,
        difficulty_match=difficulty_match,
        weakness_match=weakness_match,
        novelty_score=novelty_score,
        final_score=final_score,
    )
