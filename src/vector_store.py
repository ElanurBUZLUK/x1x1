from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Metni vektöre çevirmek için embedding modelini hazırlar."""
    return HuggingFaceEmbeddings(model_name=config.embedding_model)


def get_vector_store() -> Chroma:
    """Diskteki kalıcı Chroma veritabanını yükler."""
    return Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.vector_db_dir),
        embedding_function=get_embedding_model(),
    )
