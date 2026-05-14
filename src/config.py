from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RAGConfig:
    """Tek merkezden yönetilen proje ayarları."""

    # Eski PDF RAG ayarları korunuyor; temel projeyi bozmadık.
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    vector_db_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    collection_name: str = "pdf_rag_collection"

    # KPSS adaptive RAG ayarları.
    questions_path: Path = PROJECT_ROOT / "data" / "questions" / "kpss_questions_sample.json"
    user_history_path: Path = PROJECT_ROOT / "data" / "users" / "sample_user_history.json"
    kpss_vector_db_dir: Path = PROJECT_ROOT / "storage" / "chroma_kpss"
    kpss_collection_name: str = "kpss_question_collection"

    # Çok dilli model Türkçe soru metinleri için güvenli başlangıçtır.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Ders projesi Ollama çizgisinde olduğu için varsayılan yerel LLM.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")

    # Açık uçlu cevap puanlama için fine-tuned Transformer regression modeli.
    # Colab çıktısını varsayılan olarak buraya koy:
    # rag_project/models/answer_scoring_transformer_model/
    open_answer_transformer_model: Path = Path(
        os.getenv(
            "OPEN_ANSWER_TRANSFORMER_MODEL",
            str(PROJECT_ROOT / "models" / "answer_scoring_transformer_model"),
        )
    )
    open_answer_transformer_max_length: int = int(
        os.getenv("OPEN_ANSWER_TRANSFORMER_MAX_LENGTH", "128")
    )
    open_answer_transformer_device: str = os.getenv("OPEN_ANSWER_TRANSFORMER_DEVICE", "auto")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Spotify-benzeri soru öneri ağırlıkları.
    topic_weight: float = float(os.getenv("TOPIC_WEIGHT", "0.30"))
    difficulty_weight: float = float(os.getenv("DIFFICULTY_WEIGHT", "0.25"))
    weakness_weight: float = float(os.getenv("WEAKNESS_WEIGHT", "0.20"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.15"))
    novelty_weight: float = float(os.getenv("NOVELTY_WEIGHT", "0.10"))

    # Kullanıcı seviyesinin etrafında önerilecek zorluk bandı.
    difficulty_band: float = float(os.getenv("DIFFICULTY_BAND", "0.18"))


config = RAGConfig()
