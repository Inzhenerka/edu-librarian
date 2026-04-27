from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from edu_librarian.rag.corpus import Chunk, read_corpus_metadata, read_corpus_document


def iter_corpus_chunks(
    corpus_path: Path,
    size: int = 900,
    overlap: int = 180,
) -> Iterable[Chunk]:
    """Стримит чанки корпуса через RecursiveCharacterTextSplitter."""

    # Создаем и настраиваем сплиттер
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Перебираем все документы корпуса
    for doc in read_corpus_metadata(corpus_path):
        # Считываем текст из файла
        text = read_corpus_document(corpus_path, doc=doc)
        # Готовим словарь метаданных
        metadata = doc.model_dump()
        # Перебираем все чанки в тексте
        for ordinal, split in enumerate(splitter.create_documents([text]), start=1):
            # Извлекаем начальную позицию из метаданных чанка
            start = split.metadata["start_index"]
            # Стримим итоговый типизированный чанк
            yield Chunk(
                id=f"{doc.doc_id}::chunk_{ordinal:04d}",
                doc_id=doc.doc_id,
                chunk_text=split.page_content,
                start_pos=start,
                end_pos=start + len(split.page_content),
                metadata=metadata,
            )
