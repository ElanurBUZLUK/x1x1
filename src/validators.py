from __future__ import annotations

from collections import Counter

from src.schemas import KPSSQuestion


class QuestionBankValidationError(ValueError):
    """Soru bankası production indeksine alınamayacak kadar hatalı olduğunda kullanılır."""


def validate_question_bank(questions: list[KPSSQuestion], *, fail_on_quality_warning: bool = False) -> dict[str, object]:
    """Soru bankasını indeks öncesi kontrol eder.

    Pydantic yapısal validasyonu `KPSSQuestion` içinde yapılır. Bu fonksiyon ise
    koleksiyon seviyesinde tekrar eden ID ve kalite uyarısı gibi problemleri yakalar.
    """
    ids = [q.question_id for q in questions]
    duplicates = sorted([qid for qid, count in Counter(ids).items() if count > 1])
    warnings = [q.question_id for q in questions if q.quality_warning]

    errors: list[str] = []
    if duplicates:
        errors.append(f"Tekrarlanan question_id var: {duplicates}")
    if fail_on_quality_warning and warnings:
        errors.append(f"quality_warning içeren sorular indekslenemez: {warnings}")

    if errors:
        raise QuestionBankValidationError(" | ".join(errors))

    return {
        "question_count": len(questions),
        "duplicate_ids": duplicates,
        "quality_warning_ids": warnings,
        "quality_warning_count": len(warnings),
    }
