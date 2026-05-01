from __future__ import annotations

from src.rag_engine import PDFRAGEngine


def main() -> None:
    rag = PDFRAGEngine()
    print("PDF RAG CLI hazır. Çıkmak için 'q' yaz.")

    while True:
        question = input("\nSoru: ").strip()
        if question.lower() in {"q", "quit", "exit", "çık"}:
            break

        result = rag.ask(question)
        print("\nCevap:")
        print(result.answer)
        print("\nKaynaklar:")
        for src in result.sources:
            print(f"- {src.source} | sayfa={src.page} | {src.content_preview[:160]}...")


if __name__ == "__main__":
    main()
