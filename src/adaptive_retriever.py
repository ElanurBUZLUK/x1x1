from __future__ import annotations

import json
from typing import Any

from src.config import config
from src.embedding_store import get_kpss_vector_store
from src.question_ingest import load_questions
from src.ranking import score_candidate
from src.schemas import CandidateQuestion, KPSSQuestion, RecommendedQuestion, StudentProfile
from src.weakness_detector import choose_target_topic


class AdaptiveQuestionRetriever:
    """Spotify-benzeri KPSS soru öneri motoru.

    Retrieval tarafı Chroma'dan semantik adayları getirir.
    Ranking tarafı kullanıcının seviyesi, zayıf konusu ve çözüm geçmişiyle adayları sıralar.
    """

    def __init__(self) -> None:
        self.vector_store = get_kpss_vector_store()
        self.questions_by_id: dict[str, KPSSQuestion] = {q.question_id: q for q in load_questions()}

    def _build_query(self, profile: StudentProfile, target_topic: str | None, target_lesson: str | None) -> str:
        if target_topic:
            lesson, _, topic = target_topic.partition("/")
        else:
            lesson, topic = target_lesson or "KPSS", "genel tekrar"
        return f"{lesson} {topic} KPSS {profile.overall_level:.2f} seviyesine uygun soru"

    @staticmethod
    def _similarity_from_distance(distance: float) -> float:
        # Chroma genelde düşük mesafeyi iyi eşleşme olarak döndürür.
        # Basit ve güvenli normalizasyon: 1 / (1 + distance)
        try:
            return max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
        except Exception:
            return 0.5

    def retrieve_candidates(
        self,
        profile: StudentProfile,
        target_lesson: str | None = None,
        n_candidates: int = 30,
    ) -> tuple[str | None, list[CandidateQuestion]]:
        target_topic = choose_target_topic(profile, target_lesson=target_lesson)
        query = self._build_query(profile, target_topic=target_topic, target_lesson=target_lesson)

        docs_with_scores: list[tuple[Any, float]] = self.vector_store.similarity_search_with_score(
            query=query,
            k=n_candidates,
        )

        low = max(0.0, profile.overall_level - config.difficulty_band)
        high = min(1.0, profile.overall_level + config.difficulty_band)

        candidates: list[CandidateQuestion] = []
        for doc, distance in docs_with_scores:
            question_id = doc.metadata.get("question_id")
            if not question_id or question_id not in self.questions_by_id:
                continue

            question = self.questions_by_id[question_id]

            # Ders filtresi verilmişse dış dersleri sert şekilde ele.
            if target_lesson and question.lesson != target_lesson:
                continue

            # Çözülmüş soruyu normalde önermiyoruz; ama aday havuzu çok dar ise ranking içinde cezalandırmak için tutulabilir.
            if question.question_id in profile.solved_question_ids:
                continue

            # Çok uzak zorlukları ilk MVP'de ele.
            if not (low <= question.difficulty <= high):
                continue

            semantic_similarity = self._similarity_from_distance(distance)
            candidates.append(
                score_candidate(
                    question=question,
                    profile=profile,
                    target_topic=target_topic,
                    semantic_similarity=semantic_similarity,
                )
            )

        # Eğer filtreler çok sert olup aday bırakmadıysa önce zorluk filtresini gevşet.
        if not candidates:
            for doc, distance in docs_with_scores:
                question_id = doc.metadata.get("question_id")
                if not question_id or question_id not in self.questions_by_id:
                    continue
                question = self.questions_by_id[question_id]
                if target_lesson and question.lesson != target_lesson:
                    continue
                if question.question_id in profile.solved_question_ids:
                    continue
                semantic_similarity = self._similarity_from_distance(distance)
                candidates.append(score_candidate(question, profile, target_topic, semantic_similarity))

        # Soru bankası çok küçükse son fallback: çözülmüş soruları elemek yerine novelty=0 cezasıyla puanla.
        # Bu sayede demo tamamen kilitlenmez; production ortamında daha büyük bankada bu yola nadiren düşülür.
        if not candidates:
            for doc, distance in docs_with_scores:
                question_id = doc.metadata.get("question_id")
                if not question_id or question_id not in self.questions_by_id:
                    continue
                question = self.questions_by_id[question_id]
                if target_lesson and question.lesson != target_lesson:
                    continue
                semantic_similarity = self._similarity_from_distance(distance)
                candidates.append(score_candidate(question, profile, target_topic, semantic_similarity))

        return target_topic, sorted(candidates, key=lambda item: item.final_score, reverse=True)

    def recommend(
        self,
        profile: StudentProfile,
        target_lesson: str | None = None,
        n_candidates: int = 30,
    ) -> RecommendedQuestion:
        target_topic, candidates = self.retrieve_candidates(profile, target_lesson, n_candidates)
        if not candidates:
            raise ValueError("Uygun aday soru bulunamadı. Soru bankasını veya filtreleri genişlet.")

        best = candidates[0]
        q = best.question
        reason = (
            f"Öğrenci seviyesi {profile.overall_level:.2f}. "
            f"Hedef zayıf konu: {target_topic or 'genel tekrar'}. "
            f"Seçilen soru zorluğu {q.difficulty:.2f}; seviye-zorluk uyumu {best.difficulty_match:.2f}."
        )

        return RecommendedQuestion(
            selected_question=q,
            user_level=profile.overall_level,
            weak_topic=target_topic,
            score_breakdown={
                "topic_match": best.topic_match,
                "difficulty_match": best.difficulty_match,
                "weakness_match": best.weakness_match,
                "semantic_similarity": best.semantic_similarity,
                "novelty_score": best.novelty_score,
                "final_score": best.final_score,
            },
            selection_reason=reason,
            profile_summary=profile,
        )

    def recommend_json(self, profile: StudentProfile, target_lesson: str | None = None) -> str:
        return json.dumps(self.recommend(profile, target_lesson).model_dump(), ensure_ascii=False, indent=2)
