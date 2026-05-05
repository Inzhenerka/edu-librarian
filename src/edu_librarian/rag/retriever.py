from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from edu_librarian.config import RetrieverConfig


def get_retriever(
    config: RetrieverConfig,
    vector_store: VectorStore,
) -> VectorStoreRetriever:
    """Возвращает настроенный ретривер документов из векторной БД."""
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=config.model_dump(),
    )
