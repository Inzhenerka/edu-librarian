from langchain_openai import OpenAIEmbeddings

from edu_librarian.config import EmbedderConfig


def get_embedder(config: EmbedderConfig) -> OpenAIEmbeddings:
    """Настройка эмбеддера OpenAIEmbeddings."""
    return OpenAIEmbeddings(
        model=config.model,
        base_url=config.base_url,
        dimensions=config.dimensions,
        timeout=config.timeout,
    )
