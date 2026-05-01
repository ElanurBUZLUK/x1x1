from __future__ import annotations

from functools import lru_cache
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.adaptive_retriever import AdaptiveQuestionRetriever
from src.explanation_generator import ExplanationGenerator
from src.feedback_updater import make_answer_event
from src.open_answer_evaluator import OpenAnswerEvaluator
from src.question_ingest import build_kpss_vector_db
from src.rag_engine import PDFRAGEngine
from src.schemas import (
    KPSSQuestion,
    OpenAnswerEvaluation,
    RecommendedQuestion,
    StudentProfile,
    UserAnswerEvent,
)
from src.user_profile import append_user_answer_event, build_student_profile, load_user_history

app = FastAPI(
    title="KPSS Adaptive RAG API",
    description=(
        "Temel PDF RAG projesi üzerine kurulu; KPSS soru bankasından öğrencinin "
        "seviyesine ve zayıf konusuna uygun soru öneren AI servisi."
    ),
    version="2.1.0",
)


# -------------------------
# Eski PDF RAG endpointleri
# -------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="PDF hakkında sorulacak soru")


class SourceDocument(BaseModel):
    source: str
    page: str | int
    content_preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]


@lru_cache(maxsize=1)
def get_pdf_engine() -> PDFRAGEngine:
    return PDFRAGEngine()


# -------------------------
# KPSS adaptive endpointleri
# -------------------------
class RecommendQuestionRequest(BaseModel):
    user_id: str = "u_001"
    target_lesson: str | None = Field(default=None, description="Örn: Vatandaşlık, Tarih, Türkçe")
    history: list[UserAnswerEvent] | None = Field(
        default=None,
        description="Backend kendi kullanıcı geçmişini gönderirse burası kullanılır; boşsa sample_user_history okunur.",
    )
    rebuild_index: bool = False


class SubmitAnswerRequest(BaseModel):
    user_id: str = "u_001"
    question_id: str
    user_answer: str = Field(..., min_length=1, max_length=1)
    response_time: float | None = Field(default=None, ge=0.0)
    history: list[UserAnswerEvent] | None = None
    persist: bool = Field(
        default=True,
        description="history gönderilmediyse cevabı data/users/sample_user_history.json içine ekler.",
    )

    @field_validator("user_answer")
    @classmethod
    def validate_user_answer(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"A", "B", "C", "D", "E"}:
            raise ValueError("user_answer A/B/C/D/E olmalı.")
        return value


class SubmitAnswerResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    updated_user_level_preview: float
    updated_profile: StudentProfile


class EvaluateOpenAnswerRequest(BaseModel):
    """Serbest metin / konuşmadan metne çevrilmiş öğrenci cevabını değerlendirir.

    Şıklı cevap endpointinden ayrıdır; burada A/B/C/D/E zorunluluğu yoktur.
    """

    user_id: str = "u_001"
    question_id: str | None = None

    question_text: str = Field(..., min_length=3)
    reference_answer: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)

    grading_context: str = Field(
        default="",
        description=(
            "Cevap değerlendirmesinde kullanılacak kısa bağlam. "
            "LLM yalnızca bu bağlam ve referans cevaba dayanır."
        ),
    )

    accepted_aliases: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    wrong_if_mentions: list[str] = Field(default_factory=list)
    partial_credit_rules: list[str] = Field(default_factory=list)
    strictness_level: Literal["lenient", "normal", "strict"] = "normal"

    min_correctness_score: float = Field(default=0.65, ge=0.0, le=1.0)

    use_llm: bool = Field(
        default=True,
        description=(
            "Ollama erişilebilirse LLM-as-a-grader kullanılır; "
            "erişilemezse deterministik fallback döner."
        ),
    )

    force_llm: bool = Field(
        default=False,
        description="Exact/alias match olsa bile LLM değerlendirmesini dene.",
    )


class EvaluateOpenAnswerResponse(OpenAnswerEvaluation):
    pass


@lru_cache(maxsize=1)
def get_adaptive_retriever() -> AdaptiveQuestionRetriever:
    return AdaptiveQuestionRetriever()


@lru_cache(maxsize=1)
def get_explainer() -> ExplanationGenerator:
    return ExplanationGenerator()


@lru_cache(maxsize=2)
def get_open_answer_evaluator(use_llm: bool = True) -> OpenAnswerEvaluator:
    return OpenAnswerEvaluator(use_llm=use_llm)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"message": "KPSS Adaptive RAG servisi çalışıyor."}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Temel PDF RAG endpointi korunmuştur."""
    try:
        result = get_pdf_engine().ask(request.question)
        return AnswerResponse(
            question=result.question,
            answer=result.answer,
            sources=[SourceDocument(**src.__dict__) for src in result.sources],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {exc}") from exc


@app.post("/kpss/rebuild-index")
def rebuild_kpss_index() -> dict[str, int]:
    try:
        get_adaptive_retriever.cache_clear()
        return build_kpss_vector_db(reset=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Soru bankası validasyon hatası: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"İndeks oluşturma hatası: {exc}") from exc


@app.post("/kpss/recommend-question", response_model=RecommendedQuestion)
def recommend_question(request: RecommendQuestionRequest) -> RecommendedQuestion:
    try:
        if request.rebuild_index:
            get_adaptive_retriever.cache_clear()
            build_kpss_vector_db(reset=True)

        history = request.history if request.history is not None else load_user_history()
        profile = build_student_profile(history, user_id=request.user_id)
        return get_adaptive_retriever().recommend(profile, target_lesson=request.target_lesson)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Soru öneri hatası: {exc}") from exc


@app.post("/kpss/evaluate-open-answer", response_model=EvaluateOpenAnswerResponse)
def evaluate_open_answer(request: EvaluateOpenAnswerRequest) -> EvaluateOpenAnswerResponse:
    """Açık uçlu / tek cevaplı öğrenci cevabını referans cevaba göre değerlendirir.

    Bu endpoint klasik chatbot değildir: doğru cevabı LLM üretmez.
    LLM yalnızca verilen reference_answer + grading_context temelinde öğrencinin
    cevabını doğru/kısmen doğru/yanlış olarak yorumlar.
    """
    try:
        evaluator = get_open_answer_evaluator(request.use_llm)

        result = evaluator.evaluate(
            question_text=request.question_text,
            reference_answer=request.reference_answer,
            student_answer=request.student_answer,
            grading_context=request.grading_context,
            accepted_aliases=request.accepted_aliases,
            key_concepts=request.key_concepts,
            wrong_if_mentions=request.wrong_if_mentions,
            partial_credit_rules=request.partial_credit_rules,
            strictness_level=request.strictness_level,
            min_correctness_score=request.min_correctness_score,
            force_llm=request.force_llm,
        )

        return EvaluateOpenAnswerResponse.model_validate(result.model_dump())

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Açık cevap değerlendirme hatası: {exc}",
        ) from exc


@app.post("/kpss/submit-answer", response_model=SubmitAnswerResponse)
def submit_answer(request: SubmitAnswerRequest) -> SubmitAnswerResponse:
    try:
        history = request.history if request.history is not None else load_user_history()
        profile = build_student_profile(history, user_id=request.user_id)
        retriever = get_adaptive_retriever()
        question: KPSSQuestion | None = retriever.questions_by_id.get(request.question_id)
        if question is None:
            raise HTTPException(status_code=404, detail=f"Soru bulunamadı: {request.question_id}")

        is_correct = request.user_answer == question.correct_answer
        explanation = get_explainer().generate(question, profile, user_answer=request.user_answer)

        new_event = make_answer_event(
            user_id=request.user_id,
            question=question,
            user_answer=request.user_answer,
            response_time=request.response_time,
        )

        if request.history is None and request.persist:
            updated_history = append_user_answer_event(new_event)
        else:
            updated_history = [*history, new_event]

        updated_profile = build_student_profile(updated_history, user_id=request.user_id)

        return SubmitAnswerResponse(
            is_correct=is_correct,
            correct_answer=question.correct_answer,
            explanation=explanation,
            updated_user_level_preview=updated_profile.overall_level,
            updated_profile=updated_profile,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cevap değerlendirme hatası: {exc}") from exc
