# Open Answer Evaluator Integration

Bu ekleme, mevcut şıklı KPSS cevap kontrolünü bozmadan serbest metin / speech-to-text cevabı değerlendirmek için yapılmıştır.

## Yeni endpoint

`POST /kpss/evaluate-open-answer`

Örnek request:

```json
{
  "question_text": "Kanunları yayımlama yetkisi kime aittir?",
  "reference_answer": "Cumhurbaşkanı",
  "student_answer": "cumhur başkanı",
  "grading_context": "1982 Anayasasına göre kanunları yayımlama yetkisi Cumhurbaşkanına aittir.",
  "accepted_aliases": ["cumhurbaşkanı", "cumhur başkanı"],
  "key_concepts": ["Cumhurbaşkanı"],
  "wrong_if_mentions": ["Türkiye Büyük Millet Meclisi", "TBMM"],
  "partial_credit_rules": ["Cumhurbaşkanlığı makamını işaret eden konuşma dili cevapları kabul edilebilir."],
  "strictness_level": "normal",
  "use_llm": true
}
```

Örnek response:

```json
{
  "label": "correct",
  "is_correct": true,
  "score": 0.94,
  "confidence": 0.95,
  "normalized_student_answer": "cumhur baskani",
  "normalized_reference_answer": "cumhurbaskani",
  "matched_concepts": ["Cumhurbaşkanı"],
  "missing_concepts": [],
  "misconception": null,
  "feedback": "Doğru. Beklenen cevap: Cumhurbaşkanı.",
  "expected_answer": "Cumhurbaşkanı",
  "grading_basis": "reference_answer_and_context_only",
  "used_llm": false
}
```

## Güvenlik kuralı

LLM doğru cevabı üretmez. Yalnızca verilen `reference_answer`, `accepted_aliases`, `key_concepts` ve `grading_context` alanlarına göre öğrenci cevabını yorumlar.

## Tasarım

- Normalization
- Exact match
- Alias match
- Token / string similarity
- Key concept coverage
- Wrong-if-mentioned hard fail
- Partial credit rules for LLM grading
- Strictness level: `lenient`, `normal`, `strict`
- LLM-as-a-grader fallback
- Pydantic JSON validation

## Değişmeyen endpoint

`POST /kpss/submit-answer` hala şıklı cevaplar için çalışır: `A / B / C / D / E`.
