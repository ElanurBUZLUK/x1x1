from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_ollama import OllamaLLM

from src.config import config
from src.vector_store import get_vector_store


SYSTEM_PROMPT = """
Sen yalnızca verilen PDF/TXT bağlamına göre cevap veren bir RAG asistanısın.

Kesin kurallar:
1. Cevabı sadece BAĞLAM bölümündeki bilgilerden üret.
2. Bağlamda yeterli bilgi yoksa tahmin yürütme; "Bu bilgi verilen dokümanda bulunamadı." de.
3. Bağlamdaki bilgi kısmiyse, önce bilinen kısmı söyle, sonra eksik kısmı açıkça belirt.
4. Uydurma tarih, isim, sayı, kaynak veya yorum ekleme.
5. Cevabı Türkçe, kısa ve doğrudan ver.
6. Cevabın sonunda kullandığın kaynak etiketlerini parantez içinde belirt. Örnek: (Kaynak 1, Kaynak 3)
""".strip()


@dataclass
class SourcePreview:
    source: str
    page: str | int
    content_preview: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[SourcePreview]


def _format_context(docs: list[Any]) -> str:
    """Retriever'dan gelen chunk'ları LLM prompt'una koyulabilir metne dönüştürür."""
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "bilinmiyor")
        page = doc.metadata.get("page", "bilinmiyor")
        parts.append(f"[Kaynak {idx} | dosya={source} | sayfa={page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _make_prompt(question: str, context: str) -> str:
    return f"""
{SYSTEM_PROMPT}

BAĞLAM:
{context}

SORU:
{question}

Yanıt biçimi:
- Önce doğrudan cevabı ver.
- Gerekirse en fazla 2-3 kısa maddeyle açıkla.
- Son satırda kaynak etiketlerini yaz.

CEVAP:
""".strip()


class PDFRAGEngine:
    """Retrieval + augmentation + generation akışını tek sınıfta toplar."""

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.top_k},
        )
        self.llm = OllamaLLM(model=config.ollama_model, temperature=config.temperature)

    def ask(self, question: str) -> RAGAnswer:
        if not question.strip():
            raise ValueError("Soru boş olamaz.")

        docs = self.retriever.invoke(question)
        if not docs:
            return RAGAnswer(
                question=question,
                answer="Bu bilgi verilen dokümanda bulunamadı.",
                sources=[],
            )

        context = _format_context(docs)
        prompt = _make_prompt(question=question, context=context)
        answer = self.llm.invoke(prompt)

        sources = [
            SourcePreview(
                source=str(doc.metadata.get("source", "bilinmiyor")),
                page=doc.metadata.get("page", "bilinmiyor"),
                content_preview=doc.page_content[:350].replace("\n", " "),
            )
            for doc in docs
        ]
        return RAGAnswer(question=question, answer=str(answer).strip(), sources=sources)
