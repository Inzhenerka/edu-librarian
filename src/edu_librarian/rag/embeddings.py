from langchain_openai import OpenAIEmbeddings

from edu_librarian.config import EmbeddingsConfig


def get_embedder(config: EmbeddingsConfig) -> OpenAIEmbeddings:
    """Поднимает langchain-эмбеддер по параметрам из конфигурации.

    OpenAIEmbeddings даёт стандартный интерфейс embed_documents/embed_query,
    которым пользуются ingest и retriever.
    """
    return OpenAIEmbeddings(
        model=config.model,
        base_url=config.base_url,
        timeout=config.timeout,
    )
