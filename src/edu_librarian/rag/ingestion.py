from langchain_qdrant import QdrantVectorStore
from loguru import logger

from edu_librarian.config import RAGConfig
from edu_librarian.rag.chunk import load_chunks
from edu_librarian.rag.corpus import get_corpus_loader
from edu_librarian.rag.embedder import get_embedder
from edu_librarian.rag.text_splitter import get_text_splitter
from edu_librarian.rag.vector_store import get_vector_store


def ingest_corpus(config: RAGConfig, force: bool = False) -> QdrantVectorStore:
    """Собирает все компоненты RAG и подготавливает векторное хранилище."""
    embedder = get_embedder(config.embedder)
    vector_store, collection_already_exists = get_vector_store(config.store, embedder)
    if collection_already_exists and not force:
        logger.info(f"Reusing existing collection {vector_store.collection_name}")
        return vector_store

    loader = get_corpus_loader(config.corpus)
    splitter = get_text_splitter(config.splitter)
    chunks = load_chunks(loader, splitter)
    logger.info(f"Ingesting corpus: {loader.manifest.name}. Chunks: {len(chunks)}")
    vector_store.add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks into {vector_store.collection_name} collection")
    return vector_store
