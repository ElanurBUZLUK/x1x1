from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.schemas import OpenAnswerEvaluation

_TURKISH_CHAR_MAP = str.maketrans({
    "ç": "c",
    "ğ": "g",
    "ı": "i",
    "ö": "o",
    "ş": "s",
    "ü": "u",
    "â": "a",
    "î": "i",
    "û": "u",
})

_STOPWORDS = {
    "ve",
    "veya",
    "ile",
    "de",
    "da",
    "ki",
    "mi",
    "mu",
    "mü",
    "mı",
    "bir",
    "bu",
    "şu",
    "o",
    "olan",
    "olarak",
    "cevabım",
    "cevap",
    "bence",
    "sanırım",
    "galiba",
    "diye",
    "düşünüyorum",
    "olacak",
    "olmalı",
    "aittir",
}

_NUMBER_ALIASES = {
    "bin sekiz yuz yetmis alti": "1876",
    "bin dokuz yuz yirmi uc": "1923",
    "bin dokuz yuz yirmi dort": "1924",
    "bin dokuz yuz altmis bir": "1961",
    "bin dokuz yuz seksen iki": "1982",
}


class TransformerAnswerScorer:
    """Fine-tuned Hugging Face regression model wrapper for 0-10 answer scoring."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 128

        try:
            from src.config import config

            model_path = Path(config.open_answer_transformer_model)
            if not model_path.exists():
                return

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.max_length = config.open_answer_transformer_max_length
            if config.open_answer_transformer_device == "auto":
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_name = config.open_answer_transformer_device

            self.device = torch.device(device_name)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()

        except Exception:
            self.model = None
            self.tokenizer = None
            self.device = None

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.device is not None

    @staticmethod
    def build_input_text(question: str, reference_answer: str, student_answer: str) -> str:
        return (
            f"Soru: {question}\n"
            f"Referans Cevap: {reference_answer}\n"
            f"Öğrenci Cevabı: {student_answer}"
        )

    def predict_score_0_10(self, *, question: str, reference_answer: str, student_answer: str) -> float | None:
        if not self.is_available:
            return None

        try:
            import torch

            input_text = self.build_input_text(question, reference_answer, student_answer)
            encoding = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predicted = float(outputs.logits.squeeze().item()) * 10.0

            return max(0.0, min(10.0, predicted))

        except Exception:
            return None


class OpenAnswerEvaluator:
    """LLM destekli açık uçlu kısa cevap değerlendirici.

    LLM doğru cevabı üretmez; sadece `reference_answer`, `accepted_aliases`,
    `key_concepts` ve `grading_context` temelinde öğrenci cevabını değerlendirir.
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self.llm = None
        self.transformer_scorer = TransformerAnswerScorer()

        if use_llm:
            try:
                from langchain_ollama import OllamaLLM
                from src.config import config

                self.llm = OllamaLLM(model=config.ollama_model, temperature=0.0)
            except Exception:
                self.llm = None

    @staticmethod
    def normalize_text(text: str) -> str:
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = value.strip().lower().translate(_TURKISH_CHAR_MAP)
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return _NUMBER_ALIASES.get(value, value)

    @classmethod
    def tokenize(cls, text: str) -> set[str]:
        normalized = cls.normalize_text(text)
        return {
            token
            for token in normalized.split()
            if token and token not in _STOPWORDS
        }

    @classmethod
    def token_overlap(cls, left: str, right: str) -> float:
        left_tokens = cls.tokenize(left)
        right_tokens = cls.tokenize(right)

        if not left_tokens or not right_tokens:
            return 0.0

        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @classmethod
    def string_similarity(cls, left: str, right: str) -> float:
        left_norm = cls.normalize_text(left)
        right_norm = cls.normalize_text(right)

        if not left_norm or not right_norm:
            return 0.0

        return SequenceMatcher(None, left_norm, right_norm).ratio()

    @classmethod
    def concept_coverage(
        cls,
        student_answer: str,
        concepts: list[str],
    ) -> tuple[float, list[str], list[str]]:
        if not concepts:
            return 0.0, [], []

        normalized_student = cls.normalize_text(student_answer)
        student_tokens = cls.tokenize(normalized_student)

        matched: list[str] = []
        missing: list[str] = []

        for concept in concepts:
            normalized_concept = cls.normalize_text(concept)
            concept_tokens = cls.tokenize(normalized_concept)

            if normalized_concept and (
                normalized_concept in normalized_student
                or (concept_tokens and concept_tokens <= student_tokens)
            ):
                matched.append(concept)
            else:
                missing.append(concept)

        total = len(matched) + len(missing)
        return (len(matched) / total if total else 0.0), matched, missing

    @staticmethod
    def _label_from_score(score: float, threshold: float) -> tuple[str, bool]:
        if score >= threshold:
            return "correct", True

        if score >= max(0.45, threshold - 0.20):
            return "partially_correct", False

        return "incorrect", False

    def _deterministic_evaluate(
        self,
        *,
        reference_answer: str,
        student_answer: str,
        accepted_aliases: list[str] | None,
        key_concepts: list[str] | None,
        wrong_if_mentions: list[str] | None,
        min_correctness_score: float,
        used_llm: bool = False,
    ) -> OpenAnswerEvaluation:
        aliases = accepted_aliases or []
        concepts = key_concepts or []

        normalized_student = self.normalize_text(student_answer)
        normalized_reference = self.normalize_text(reference_answer)
        normalized_aliases = [self.normalize_text(alias) for alias in aliases]

        wrong_hits = [
            wrong
            for wrong in (wrong_if_mentions or [])
            if self.normalize_text(wrong) in normalized_student
        ]

        if wrong_hits:
            return OpenAnswerEvaluation(
                label="incorrect",
                is_correct=False,
                score=0.05,
                confidence=0.95,
                normalized_student_answer=normalized_student,
                normalized_reference_answer=normalized_reference,
                matched_concepts=[],
                missing_concepts=concepts,
                misconception=f"Öğrenci cevabı yanlış kabul edilen kavram içeriyor: {', '.join(wrong_hits)}",
                feedback=f"Yanlış. Beklenen cevap: {reference_answer}.",
                expected_answer=reference_answer,
                grading_basis="reference_answer_and_context_only",
                used_llm=used_llm,
            )

        exact_match = normalized_student == normalized_reference and bool(normalized_student)
        alias_match = normalized_student in normalized_aliases and bool(normalized_student)
        ref_in_student = bool(normalized_reference) and normalized_reference in normalized_student
        student_in_ref = bool(normalized_student) and normalized_student in normalized_reference

        lexical_similarity = self.string_similarity(student_answer, reference_answer)
        overlap = self.token_overlap(student_answer, reference_answer)
        coverage, matched, missing = self.concept_coverage(student_answer, concepts)

        if exact_match or alias_match or ref_in_student:
            score = 0.98 if exact_match else 0.94
            confidence = 0.95

        elif student_in_ref and len(normalized_student) >= 4:
            score = 0.82
            confidence = 0.80

        else:
            score = (
                0.45 * lexical_similarity
                + 0.35 * overlap
                + 0.20 * coverage
            )
            confidence = min(
                0.90,
                0.40
                + 0.30 * coverage
                + 0.20 * overlap
                + 0.10 * lexical_similarity,
            )

            if not concepts:
                confidence = min(0.75, confidence)

        score = max(0.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))

        label, is_correct = self._label_from_score(score, min_correctness_score)

        if label == "correct":
            feedback = f"Doğru. Beklenen cevap: {reference_answer}."
        elif label == "partially_correct":
            feedback = (
                f"Kısmen doğru olabilir; beklenen cevapla tam örtüşmüyor. "
                f"Beklenen cevap: {reference_answer}."
            )
        else:
            feedback = f"Yanlış. Beklenen cevap: {reference_answer}."

        return OpenAnswerEvaluation(
            label=label,
            is_correct=is_correct,
            score=round(score, 4),
            confidence=round(confidence, 4),
            normalized_student_answer=normalized_student,
            normalized_reference_answer=normalized_reference,
            matched_concepts=matched,
            missing_concepts=missing,
            misconception=None,
            feedback=feedback,
            expected_answer=reference_answer,
            grading_basis="reference_answer_and_context_only",
            used_llm=used_llm,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _llm_evaluate(
        self,
        *,
        question_text: str,
        reference_answer: str,
        student_answer: str,
        grading_context: str,
        accepted_aliases: list[str] | None,
        key_concepts: list[str] | None,
        wrong_if_mentions: list[str] | None,
        partial_credit_rules: list[str] | None,
        strictness_level: str,
        min_correctness_score: float,
    ) -> OpenAnswerEvaluation | None:
        if self.llm is None:
            return None

        aliases_text = ", ".join(accepted_aliases or []) or "Yok"
        concepts_text = ", ".join(key_concepts or []) or "Yok"
        wrong_text = ", ".join(wrong_if_mentions or []) or "Yok"
        partial_rules_text = "\n".join(f"- {rule}" for rule in (partial_credit_rules or [])) or "Yok"

        prompt = f"""
Sen otomatik kısa cevap değerlendirme motorusun. Sadece verilen referans cevap ve bağlama göre karar ver.

Kesin kurallar:
1. Yeni bilgi uydurma.
2. Doğru cevabı kendi genel bilginle tahmin etme.
3. Sadece REFERENCE_ANSWER, ACCEPTED_ALIASES, KEY_CONCEPTS ve GRADING_CONTEXT alanlarını ölçüt al.
4. Öğrenci cevabı konuşma diliyle yazılmış olabilir; eş anlamlı ve çekimli ifadeleri dikkate al.
5. Tam doğru değilse ama ana fikrin bir kısmını karşılıyorsa partially_correct kullan.
6. Emin değilsen confidence değerini düşür ve uncertain kullanabilirsin.
7. Öğrenci cevabı WRONG_IF_MENTIONS listesindeki kavramlardan birini ana cevap olarak içeriyorsa incorrect ver.
8. STRICTNESS_LEVEL strict ise daha az toleranslı, lenient ise daha toleranslı değerlendir.
9. Sadece geçerli JSON döndür. Markdown yazma.

QUESTION:
{question_text}

REFERENCE_ANSWER:
{reference_answer}

ACCEPTED_ALIASES:
{aliases_text}

KEY_CONCEPTS:
{concepts_text}

WRONG_IF_MENTIONS:
{wrong_text}

PARTIAL_CREDIT_RULES:
{partial_rules_text}

STRICTNESS_LEVEL:
{strictness_level}

GRADING_CONTEXT:
{grading_context or reference_answer}

STUDENT_ANSWER:
{student_answer}

JSON şeması:
{{
  "label": "correct|partially_correct|incorrect|uncertain",
  "is_correct": true,
  "score": 0.0,
  "confidence": 0.0,
  "normalized_student_answer": "...",
  "normalized_reference_answer": "...",
  "matched_concepts": [],
  "missing_concepts": [],
  "misconception": null,
  "feedback": "Kısa Türkçe geri bildirim.",
  "expected_answer": "{reference_answer}",
  "grading_basis": "reference_answer_and_context_only",
  "used_llm": true
}}
""".strip()

        try:
            raw = str(self.llm.invoke(prompt)).strip()
            payload = self._extract_json(raw)

            payload["score"] = max(0.0, min(1.0, float(payload.get("score", 0.0))))
            payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
            payload["expected_answer"] = reference_answer
            payload["grading_basis"] = "reference_answer_and_context_only"
            payload["used_llm"] = True

            if payload.get("label") == "correct" and payload["score"] < min_correctness_score:
                payload["label"] = "partially_correct"
                payload["is_correct"] = False

            return OpenAnswerEvaluation.model_validate(payload)

        except Exception:
            return None

    def _transformer_evaluate(
        self,
        *,
        question_text: str,
        reference_answer: str,
        student_answer: str,
        key_concepts: list[str] | None,
        min_correctness_score: float,
    ) -> OpenAnswerEvaluation | None:
        predicted_0_10 = self.transformer_scorer.predict_score_0_10(
            question=question_text,
            reference_answer=reference_answer,
            student_answer=student_answer,
        )
        if predicted_0_10 is None:
            return None

        score = round(predicted_0_10 / 10.0, 4)
        label, is_correct = self._label_from_score(score, min_correctness_score)
        normalized_student = self.normalize_text(student_answer)
        normalized_reference = self.normalize_text(reference_answer)
        coverage, matched, missing = self.concept_coverage(student_answer, key_concepts or [])

        distance_from_threshold = abs(score - min_correctness_score)
        confidence = 0.65 + min(0.25, distance_from_threshold)
        if coverage > 0:
            confidence = max(confidence, min(0.90, 0.65 + 0.20 * coverage))

        if label == "correct":
            feedback = f"Doğru. Transformer puanı: {predicted_0_10:.1f}/10."
        elif label == "partially_correct":
            feedback = (
                f"Kısmen doğru olabilir. Transformer puanı: {predicted_0_10:.1f}/10. "
                f"Beklenen cevap: {reference_answer}."
            )
        else:
            feedback = (
                f"Yanlış ya da yetersiz. Transformer puanı: {predicted_0_10:.1f}/10. "
                f"Beklenen cevap: {reference_answer}."
            )

        return OpenAnswerEvaluation(
            label=label,
            is_correct=is_correct,
            score=score,
            confidence=round(max(0.0, min(0.90, confidence)), 4),
            normalized_student_answer=normalized_student,
            normalized_reference_answer=normalized_reference,
            matched_concepts=matched,
            missing_concepts=missing,
            misconception=None,
            feedback=feedback,
            expected_answer=reference_answer,
            grading_basis="transformer_regression_reference_answer_and_context",
            used_llm=False,
        )

    def evaluate(
        self,
        *,
        question_text: str,
        reference_answer: str,
        student_answer: str,
        grading_context: str = "",
        accepted_aliases: list[str] | None = None,
        key_concepts: list[str] | None = None,
        wrong_if_mentions: list[str] | None = None,
        partial_credit_rules: list[str] | None = None,
        strictness_level: str = "normal",
        min_correctness_score: float = 0.65,
        force_llm: bool = False,
    ) -> OpenAnswerEvaluation:
        if not reference_answer.strip():
            raise ValueError("reference_answer boş olamaz.")

        if not student_answer.strip():
            raise ValueError("student_answer boş olamaz.")

        if not 0.0 <= min_correctness_score <= 1.0:
            raise ValueError("min_correctness_score 0.0-1.0 arasında olmalı.")

        if strictness_level not in {"lenient", "normal", "strict"}:
            raise ValueError("strictness_level lenient/normal/strict olmalı.")

        deterministic = self._deterministic_evaluate(
            reference_answer=reference_answer,
            student_answer=student_answer,
            accepted_aliases=accepted_aliases,
            key_concepts=key_concepts,
            wrong_if_mentions=wrong_if_mentions,
            min_correctness_score=min_correctness_score,
        )

        if deterministic.misconception and deterministic.score <= 0.05:
            return deterministic

        if deterministic.score >= 0.94 and deterministic.confidence >= 0.90 and not force_llm:
            return deterministic

        transformer_result = self._transformer_evaluate(
            question_text=question_text,
            reference_answer=reference_answer,
            student_answer=student_answer,
            key_concepts=key_concepts,
            min_correctness_score=min_correctness_score,
        )
        if transformer_result is not None and not force_llm:
            return transformer_result

        llm_result = self._llm_evaluate(
            question_text=question_text,
            reference_answer=reference_answer,
            student_answer=student_answer,
            grading_context=grading_context,
            accepted_aliases=accepted_aliases,
            key_concepts=key_concepts,
            wrong_if_mentions=wrong_if_mentions,
            partial_credit_rules=partial_credit_rules,
            strictness_level=strictness_level,
            min_correctness_score=min_correctness_score,
        )

        return llm_result or deterministic
