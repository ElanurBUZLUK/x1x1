from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

OPTION_KEYS = {"A", "B", "C", "D", "E"}


class KPSSQuestion(BaseModel):
    """Soru bankasındaki tek bir KPSS sorusunun normalize edilmiş şeması.

    Geriye uyumluluk korunur: eski JSON kayıtlarında eksik olan `exam`, `source`,
    `tags`, `quality_warning` gibi alanlar default değerlerle tamamlanır.
    """

    question_id: str = Field(..., min_length=1)
    exam: str = "KPSS"
    section: str = Field(..., min_length=1)
    lesson: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    subtopic: str | None = None
    difficulty: float = Field(..., ge=0.0, le=1.0)
    question_text: str = Field(..., min_length=5)
    options: dict[str, str]
    correct_answer: str
    explanation: str = Field(..., min_length=3)
    tags: list[str] = Field(default_factory=list)
    source: str = "kpss_question_bank"
    quality_warning: str | None = None

    @field_validator("correct_answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in OPTION_KEYS:
            raise ValueError("correct_answer A/B/C/D/E olmalı.")
        return value

    @field_validator("options")
    @classmethod
    def normalize_options(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {str(k).strip().upper(): str(v).strip() for k, v in value.items()}
        if set(normalized.keys()) != OPTION_KEYS:
            raise ValueError("options tam olarak A/B/C/D/E şıklarını içermeli.")
        empty = [key for key, text in normalized.items() if not text]
        if empty:
            raise ValueError(f"Boş şık metni var: {empty}")
        return normalized

    @model_validator(mode="after")
    def validate_answer_in_options(self) -> "KPSSQuestion":
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer options içinde bulunmalı.")
        return self

    def topic_key(self) -> str:
        return f"{self.lesson}/{self.topic}"

    def embedding_text(self) -> str:
        """Embedding için semantik temsil metni üretir."""
        options_text = " ".join(f"{key}) {val}" for key, val in sorted(self.options.items()))
        tags_text = ", ".join(self.tags)
        return (
            f"Sınav: {self.exam}\n"
            f"Bölüm: {self.section}\n"
            f"Ders: {self.lesson}\n"
            f"Konu: {self.topic}\n"
            f"Alt konu: {self.subtopic or ''}\n"
            f"Zorluk: {self.difficulty}\n"
            f"Etiketler: {tags_text}\n"
            f"Soru: {self.question_text}\n"
            f"Şıklar: {options_text}\n"
            f"Açıklama: {self.explanation}"
        )


class UserAnswerEvent(BaseModel):
    """Öğrencinin geçmişte çözdüğü bir soru kaydı."""

    user_id: str = "u_001"
    question_id: str = Field(..., min_length=1)
    lesson: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    difficulty: float = Field(..., ge=0.0, le=1.0)
    is_correct: bool
    response_time: float | None = Field(default=None, ge=0.0)


class StudentProfile(BaseModel):
    user_id: str
    overall_level: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    recent_accuracy: float = Field(..., ge=0.0, le=1.0)
    avg_solved_difficulty: float = Field(..., ge=0.0, le=1.0)
    topic_mastery: dict[str, float]
    weak_topics: list[str]
    solved_question_ids: list[str]


class CandidateQuestion(BaseModel):
    question: KPSSQuestion
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    topic_match: float = Field(..., ge=0.0, le=1.0)
    difficulty_match: float = Field(..., ge=0.0, le=1.0)
    weakness_match: float = Field(..., ge=0.0, le=1.0)
    novelty_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


class RecommendedQuestion(BaseModel):
    selected_question: KPSSQuestion
    user_level: float
    weak_topic: str | None
    score_breakdown: dict[str, float]
    selection_reason: str
    profile_summary: StudentProfile | None = None


class OpenAnswerEvaluation(BaseModel):
    """Açık uçlu / tek cevaplı öğrenci cevabı değerlendirme sonucu."""

    label: Literal["correct", "partially_correct", "incorrect", "uncertain"]
    is_correct: bool
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    normalized_student_answer: str
    normalized_reference_answer: str
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    misconception: str | None = None
    feedback: str
    expected_answer: str
    grading_basis: str = "reference_answer_and_context_only"
    used_llm: bool = False

    @model_validator(mode="after")
    def keep_label_consistent(self) -> "OpenAnswerEvaluation":
        if self.label == "correct" and not self.is_correct:
            raise ValueError("label='correct' ise is_correct=True olmalı.")

        if self.label in {"incorrect", "uncertain"} and self.is_correct:
            raise ValueError("incorrect/uncertain etiketlerinde is_correct=False olmalı.")

        return self
