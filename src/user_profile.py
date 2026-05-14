from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.config import config
from src.schemas import AnswerHistoryEvent, OpenAnswerAttemptEvent, StudentProfile, UserAnswerEvent


def _parse_history_event(item: dict) -> AnswerHistoryEvent:
    if item.get("answer_type") == "open_answer":
        return OpenAnswerAttemptEvent.model_validate(item)
    return UserAnswerEvent.model_validate(item)


def load_user_history(path: Path | None = None) -> list[AnswerHistoryEvent]:
    """Örnek kullanıcı çözüm geçmişini okur."""
    history_path = path or config.user_history_path
    if not history_path.exists():
        return []
    raw = json.loads(history_path.read_text(encoding="utf-8"))
    return [_parse_history_event(item) for item in raw]


def save_user_history(history: list[AnswerHistoryEvent], path: Path | None = None) -> None:
    """Çözüm geçmişini JSON dosyasına yazar. Demo backend için basit kalıcılık sağlar."""
    history_path = path or config.user_history_path
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [event.model_dump() for event in history]
    history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_answer_history_event(event: AnswerHistoryEvent, path: Path | None = None) -> list[AnswerHistoryEvent]:
    """Yeni cevap olayını geçmişe ekler ve güncel geçmişi döndürür."""
    history = load_user_history(path)
    history.append(event)
    save_user_history(history, path)
    return history


def append_user_answer_event(event: UserAnswerEvent, path: Path | None = None) -> list[AnswerHistoryEvent]:
    """Geriye uyumlu çoktan seçmeli cevap ekleme yardımcısı."""
    return append_answer_history_event(event, path)


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _event_score(event: AnswerHistoryEvent) -> float:
    if isinstance(event, OpenAnswerAttemptEvent):
        return event.answer_score
    return 1.0 if event.is_correct else 0.0


def build_student_profile(history: list[AnswerHistoryEvent], user_id: str = "u_001") -> StudentProfile:
    """Spotify'daki kullanıcı zevki yerine öğrencinin öğrenme profilini çıkarır."""
    user_events = [event for event in history if event.user_id == user_id]

    if not user_events:
        return StudentProfile(
            user_id=user_id,
            overall_level=0.35,
            accuracy=0.0,
            recent_accuracy=0.0,
            avg_solved_difficulty=0.35,
            topic_mastery={},
            weak_topics=[],
            solved_question_ids=[],
        )

    scores = [_event_score(event) for event in user_events]
    recent = scores[-10:]
    difficulties = [event.difficulty for event in user_events]

    accuracy = _safe_mean(scores)
    recent_accuracy = _safe_mean(recent)
    avg_difficulty = _safe_mean(difficulties, default=0.35)

    # Seviye sadece doğruluk değil, çözülen soru zorluğu ile birlikte değerlendirilir.
    overall_level = max(0.0, min(1.0, 0.50 * accuracy + 0.30 * recent_accuracy + 0.20 * avg_difficulty))

    topic_events: dict[str, list[AnswerHistoryEvent]] = defaultdict(list)
    for event in user_events:
        topic_key = f"{event.lesson}/{event.topic}"
        topic_events[topic_key].append(event)

    topic_mastery: dict[str, float] = {}
    for topic_key, events in topic_events.items():
        topic_accuracy = _safe_mean([_event_score(event) for event in events])
        topic_difficulty = _safe_mean([event.difficulty for event in events], default=0.35)
        mastery = max(0.0, min(1.0, 0.70 * topic_accuracy + 0.30 * topic_difficulty))
        topic_mastery[topic_key] = mastery

    weak_topics = [topic for topic, mastery in sorted(topic_mastery.items(), key=lambda item: item[1])]

    # Aynı soru birden fazla kez çözülmüşse tekrarları silerek öneri filtresini kararlı tutuyoruz.
    solved_unique = list(dict.fromkeys(event.question_id for event in user_events))

    return StudentProfile(
        user_id=user_id,
        overall_level=overall_level,
        accuracy=accuracy,
        recent_accuracy=recent_accuracy,
        avg_solved_difficulty=avg_difficulty,
        topic_mastery=topic_mastery,
        weak_topics=weak_topics,
        solved_question_ids=solved_unique,
    )
