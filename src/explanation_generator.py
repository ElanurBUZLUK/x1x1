from __future__ import annotations

from langchain_ollama import OllamaLLM

from src.config import config
from src.level_estimator import level_label
from src.schemas import KPSSQuestion, StudentProfile


class ExplanationGenerator:
    """Seçilen soru için LLM tabanlı açıklama/geri bildirim üretir.

    Not: LLM soru üretmez. Güvenli yaklaşım olarak mevcut soru bankasındaki soru ve doğru cevap üzerinden açıklama üretir.
    """

    def __init__(self) -> None:
        self.llm = OllamaLLM(model=config.ollama_model, temperature=config.temperature)

    @staticmethod
    def _fallback_explanation(question: KPSSQuestion, user_answer: str | None = None) -> str:
        if user_answer:
            correctness = "doğru" if user_answer.upper() == question.correct_answer else "yanlış"
            return (
                f"Cevabın: {user_answer.upper()} ({correctness}). Doğru cevap: {question.correct_answer}.\n"
                f"Açıklama: {question.explanation}"
            )
        return f"Doğru cevap: {question.correct_answer}. Açıklama: {question.explanation}"

    def generate(self, question: KPSSQuestion, profile: StudentProfile, user_answer: str | None = None) -> str:
        options = "\n".join(f"{key}) {value}" for key, value in sorted(question.options.items()))
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

Şıklar:
{options}

Doğru cevap: {question.correct_answer}
Hazır açıklama: {question.explanation}
Öğrenci cevabı: {user_answer or 'henüz cevaplamadı'}

Görev:
1. Doğru cevabı açıkla.
2. Öğrenci yanlış cevapladıysa neden yanılmış olabileceğini belirt.
3. Bir sonraki mikro çalışma önerisini ver.
4. Cevabı Türkçe ve kısa tut.
""".strip()
        try:
            return str(self.llm.invoke(prompt)).strip()
        except Exception:
            return self._fallback_explanation(question, user_answer)
