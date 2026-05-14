from __future__ import annotations

from langchain_ollama import OllamaLLM

from src.config import config
from src.level_estimator import level_label
from src.schemas import KPSSQuestion, StudentProfile


class ExplanationGenerator:
    """Seçilen soru için LLM tabanlı açıklama/geri bildirim üretir.

    Not: LLM soru üretmez. Güvenli yaklaşım olarak mevcut soru bankasındaki soru ve referans cevap üzerinden açıklama üretir.
    """

    def __init__(self) -> None:
        self.llm = OllamaLLM(model=config.ollama_model, temperature=config.temperature)

    @staticmethod
    def _fallback_explanation(question: KPSSQuestion, student_answer: str | None = None) -> str:
        base = f"Referans cevap: {question.reference_answer}. Açıklama: {question.explanation}"
        if student_answer:
            return f"Öğrenci cevabı: {student_answer}\n{base}"
        return base

    def generate(self, question: KPSSQuestion, profile: StudentProfile, student_answer: str | None = None) -> str:
        concepts = ", ".join(question.key_concepts) or "-"
        prompt = f"""
Sen KPSS öğrencisine kısa, net ve öğretici geri bildirim veren bir öğretmensin.
Yeni soru üretme. Sadece verilen soruyu ve açıklamayı kullan.

Öğrenci seviyesi: {level_label(profile.overall_level)} ({profile.overall_level:.2f})
Zayıf konular: {', '.join(profile.weak_topics[:3]) or 'henüz yok'}

Ders: {question.lesson}
Konu: {question.topic}
Alt konu: {question.subtopic or '-'}
Zorluk: {question.difficulty:.2f}

Soru:
{question.question_text}

Referans cevap: {question.reference_answer}
Ana kavramlar: {concepts}
Hazır açıklama: {question.explanation}
Öğrenci cevabı: {student_answer or 'henüz cevaplamadı'}

Görev:
1. Referans cevabı açıkla.
2. Öğrenci cevabı varsa eksik veya güçlü yanlarını belirt.
3. Bir sonraki mikro çalışma önerisini ver.
4. Cevabı Türkçe ve kısa tut.
""".strip()
        try:
            return str(self.llm.invoke(prompt)).strip()
        except Exception:
            return self._fallback_explanation(question, student_answer)
