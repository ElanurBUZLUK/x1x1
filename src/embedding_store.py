from __future__ import annotations

from langchain_chroma import Chroma

from src.config import config
from src.vector_store import get_embedding_model


def get_kpss_vector_store() -> Chroma:
    """KPSS soru koleksiyonunu yükler."""
    return Chroma(
        collection_name=config.kpss_collection_name,
        persist_directory=str(config.kpss_vector_db_dir),
        embedding_function=get_embedding_model(),
    )
