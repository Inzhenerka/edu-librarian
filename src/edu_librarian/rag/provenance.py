import re

from langchain_core.documents import Document

from edu_librarian.rag.chunk import ChunkMetadata

# Готовим парсер отсылок формата [N]
_REF_RE = re.compile(r"\[(\d+)]")


def parse_referenced_chunks(content: str, chunks: list[Document]) -> dict[int, ChunkMetadata]:
    """Парсит номер отсылки [N] из текста ответа, подбирает метаданные чанков, сохраняет порядок появления."""

    # Готовим выходной словарь, где номер отсылки связан с метаданными чанка
    sources: dict[int, ChunkMetadata] = {}

    # Перебор всех совпадений в тексте
    for match in _REF_RE.finditer(content):
        # Извлекаем номер отсылки из текста
        ref_num = int(match.group(1))

        # Проверяем, что номер находится в диапазоне чанков и еще не был обработан
        if 1 <= ref_num <= len(chunks) and ref_num not in sources:
            # Извлекаем чанк из массива по номеру отсылки
            # Тут важно знать, что при формировании номеров в промпте, чанки рендерились в том же порядке
            chunk = chunks[ref_num - 1]

            # Готовим и назначаем метаданные по номеру отсылки
            sources[ref_num] = ChunkMetadata.from_document(chunk)

    return sources
