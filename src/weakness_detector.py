from __future__ import annotations

from src.schemas import StudentProfile


def choose_target_topic(profile: StudentProfile, target_lesson: str | None = None) -> str | None:
    """En zayıf konu seçilir; ders filtresi verilirse o derse öncelik verilir."""
    if not profile.weak_topics:
        return None

    if target_lesson:
        prefix = f"{target_lesson}/"
        for topic in profile.weak_topics:
            if topic.startswith(prefix):
                return topic

    return profile.weak_topics[0]
