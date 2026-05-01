from __future__ import annotations

from src.schemas import StudentProfile


def level_label(score: float) -> str:
    """0-1 skorunu insan okunur seviyeye çevirir."""
    if score < 0.40:
        return "başlangıç"
    if score < 0.70:
        return "orta"
    return "ileri"


def target_difficulty_band(profile: StudentProfile, band: float = 0.18) -> tuple[float, float]:
    """Öğrencinin gelişim bandına yakın zorluk aralığını döndürür."""
    low = max(0.0, profile.overall_level - band)
    high = min(1.0, profile.overall_level + band)
    return low, high
