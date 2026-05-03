from typing import Self
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from pydantic import Field

from edu_librarian.rag.corpus import CorpusDocument, CorpusLoader


class ChunkMetadata(CorpusDocument):
    """Метаданные чанка: паспорт документа, позиция фрагмента, уникальный id."""
    start_index: int
    chunk_id: str = Field(
        default_factory=lambda data: f"{data['doc_id']}::{data['start_index']:08d}"
    )

    @classmethod
    def from_document(cls, document: Document) -> Self:
        return cls.model_validate(document.metadata)

    def apply_to_document(self, document: Document) -> None:
        document.metadata = self.model_dump()
        document.id = self.make_point_id()

    def make_point_id(self) -> str:
        """Воспроизводимый технический id для векторной базы Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.chunk_id))


def load_chunks(loader: CorpusLoader, splitter: TextSplitter) -> list[Document]:
    """Загружаем корпус, режем на чанки, назначаем идентификаторы."""
    chunks = splitter.split_documents(loader.lazy_load())
    for chunk in chunks:
        # Формализуем метаданные чанка и генерируем id
        meta = ChunkMetadata.from_document(chunk)
        # Применяем обновленные метаданные к чанку
        meta.apply_to_document(chunk)
    return chunks
