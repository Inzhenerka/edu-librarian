from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from edu_librarian.config import SplitterConfig


def get_text_splitter(config: SplitterConfig) -> TextSplitter:
    """Создание рекурсивного сплиттера документа с кастомным списком разделителей."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
