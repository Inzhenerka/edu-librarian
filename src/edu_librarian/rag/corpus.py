from pathlib import Path

import yaml
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Паспорт исходного документа корпуса. Не векторизуется."""
    doc_id: str
    file: str
    title: str
    author: str | None = None
    source: str | None = None
    source_url: str | None = None
    section: str | None = None
    rights: str | None = None
    content_scope: str | None = None


class Chunk(BaseModel):
    """Минимальная адресуемая единица знаний для векторной БД."""
    id: str
    doc_id: str
    chunk_text: str
    start_pos: int
    end_pos: int
    metadata: dict


def read_corpus_metadata(corpus_path: Path) -> list[DocumentMetadata]:
    """Загрузка паспортов корпуса."""
    raw_list: list[dict] = yaml.safe_load((corpus_path / "corpus.yml").read_text(encoding="utf-8"))
    return [DocumentMetadata.model_validate(item) for item in raw_list]


def read_corpus_document(corpus_path: Path, doc: DocumentMetadata) -> str:
    """Поиск документа по метаданным и загрузка его содержимого."""
    return (corpus_path / doc.file).read_text(encoding="utf-8").strip()
