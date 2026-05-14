from __future__ import annotations

import argparse

from src.adaptive_retriever import AdaptiveQuestionRetriever
from src.question_ingest import build_kpss_vector_db
from src.user_profile import build_student_profile, load_user_history


def main() -> None:
    parser = argparse.ArgumentParser(description="KPSS adaptif soru öneri demo scripti.")
    parser.add_argument("--user-id", default="u_001")
    parser.add_argument("--lesson", default=None, help="Örn: Vatandaşlık, Tarih, Türkçe")
    parser.add_argument("--rebuild", action="store_true", help="Önce Chroma soru indeksini yeniden oluştur.")
    args = parser.parse_args()

    if args.rebuild:
        print(build_kpss_vector_db(reset=True))

    history = load_user_history()
    profile = build_student_profile(history, user_id=args.user_id)
    recommendation = AdaptiveQuestionRetriever().recommend(profile, target_lesson=args.lesson)

    q = recommendation.selected_question
    print("\n=== ÖĞRENCİ PROFİLİ ===")
    print(f"Kullanıcı: {profile.user_id}")
    print(f"Genel seviye: {profile.overall_level:.2f}")
    print(f"Zayıf konular: {profile.weak_topics}")

    print("\n=== ÖNERİLEN SORU ===")
    print(f"ID: {q.question_id}")
    print(f"Ders/Konu: {q.lesson} / {q.topic}")
    print(f"Zorluk: {q.difficulty:.2f}")
    print(q.question_text)
    print(f"Referans cevap: {q.reference_answer}")
    if q.key_concepts:
        print(f"Ana kavramlar: {', '.join(q.key_concepts)}")

    print("\n=== NEDEN BU SORU? ===")
    print(recommendation.selection_reason)
    print(recommendation.score_breakdown)


if __name__ == "__main__":
    main()
