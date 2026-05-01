from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.vector_store import get_embedding_model


def discover_documents(input_dir: Path) -> list[Path]:
    """Desteklenen dosyaları bulur. Şimdilik PDF ve TXT yeterli."""
    supported_suffixes = {".pdf", ".txt"}
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def load_file(path: Path) -> list[Document]:
    """Dosya tipine göre loader seçer."""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    if path.suffix.lower() == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Desteklenmeyen dosya tipi: {path}")


def load_documents(paths: Iterable[Path]) -> list[Document]:
    """Tüm belgeleri LangChain Document formatına çevirir."""
    documents: list[Document] = []
    for path in paths:
        loaded = load_file(path)
        for doc in loaded:
            doc.metadata["source"] = str(path.name)
        documents.extend(loaded)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Uzun metni retrieval için küçük ve örtüşmeli chunk'lara böler."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_db(reset: bool = True) -> dict[str, int]:
    """Belgeleri okur, chunk'lara böler, embedding çıkarır ve Chroma'ya kaydeder."""
    config.vector_db_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_documents(config.raw_data_dir)
    if not paths:
        raise FileNotFoundError(
            f"{config.raw_data_dir} içinde PDF/TXT bulunamadı. Önce data/raw klasörüne belge koy."
        )

    if reset:
        # Eski indeksle yeni belge karışmasın diye koleksiyonu sıfırlıyoruz.
        vector_store = Chroma(
            collection_name=config.collection_name,
            persist_directory=str(config.vector_db_dir),
            embedding_function=get_embedding_model(),
        )
        try:
            vector_store.delete_collection()
        except Exception:
            pass

    documents = load_documents(paths)
    chunks = split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=config.collection_name,
        persist_directory=str(config.vector_db_dir),
    )

    return {"files": len(paths), "pages_or_docs": len(documents), "chunks": len(chunks)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF/TXT belgelerinden Chroma vektör veritabanı oluşturur.")
    parser.add_argument("--no-reset", action="store_true", help="Mevcut koleksiyonu silmeden ekleme yapar.")
    args = parser.parse_args()

    stats = build_vector_db(reset=not args.no_reset)
    print("Vektör veritabanı oluşturuldu:")
    print(stats)


if __name__ == "__main__":
    main()
