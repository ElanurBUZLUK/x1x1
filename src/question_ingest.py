from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import config
from src.schemas import KPSSQuestion
from src.validators import validate_question_bank
from src.vector_store import get_embedding_model


def load_questions(path: Path | None = None) -> list[KPSSQuestion]:
    """JSON soru bankasını okur ve şema validasyonundan geçirir."""
    question_path = path or config.questions_path
    if not question_path.exists():
        raise FileNotFoundError(f"Soru bankası bulunamadı: {question_path}")

    raw = json.loads(question_path.read_text(encoding="utf-8"))
    return [KPSSQuestion.model_validate(item) for item in raw]


def question_to_document(question: KPSSQuestion) -> Document:
    """Her soruyu tek bir retrieval dokümanı/chunk olarak saklar."""
    return Document(
        page_content=question.embedding_text(),
        metadata={
            "question_id": question.question_id,
            "exam": question.exam,
            "section": question.section,
            "lesson": question.lesson,
            "topic": question.topic,
            "subtopic": question.subtopic or "",
            "difficulty": question.difficulty,
            "correct_answer": question.correct_answer,
            "source": question.source,
            "quality_warning": question.quality_warning or "",
        },
    )


def build_kpss_vector_db(reset: bool = True) -> dict[str, int]:
    """KPSS soru bankasını embedding + Chroma koleksiyonuna dönüştürür."""
    questions = load_questions(config.questions_path)
    validation = validate_question_bank(questions, fail_on_quality_warning=False)
    documents = [question_to_document(question) for question in questions]

    if reset and config.kpss_vector_db_dir.exists():
        shutil.rmtree(config.kpss_vector_db_dir)

    config.kpss_vector_db_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        collection_name=config.kpss_collection_name,
        persist_directory=str(config.kpss_vector_db_dir),
    )

    warning_count = int(validation["quality_warning_count"])
    return {
        "questions": len(questions),
        "documents": len(documents),
        "quality_warnings": warning_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KPSS soru bankasını Chroma vektör veritabanına aktarır.")
    parser.add_argument("--no-reset", action="store_true", help="Mevcut koleksiyonu silmeden ekleme yapar.")
    args = parser.parse_args()

    stats = build_kpss_vector_db(reset=not args.no_reset)
    print("KPSS soru vektör veritabanı oluşturuldu:")
    print(stats)


if __name__ == "__main__":
    main()
