from pathlib import Path
from typing import Iterator, Self

import yaml
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import BaseModel

from edu_librarian.config import CorpusConfig


class CorpusDocument(BaseModel):
    """Паспорт документа корпуса."""
    doc_id: str
    file: str
    title: str
    author: str | None = None
    source: str | None = None
    source_url: str | None = None
    section: str | None = None
    content_scope: str | None = None


class CorpusManifest(BaseModel):
    """Манифест корпуса документов."""
    name: str
    description: str
    documents: list[CorpusDocument]

    @classmethod
    def from_yaml_file(cls, manifest_path: str | Path) -> Self:
        text = Path(manifest_path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.safe_load(text))


class CorpusLoader(BaseLoader):
    """Загрузчик документов корпуса с паспортами из манифеста."""

    def __init__(self, config: CorpusConfig):
        self.text_dir = config.text_dir
        self.manifest = CorpusManifest.from_yaml_file(config.manifest_file)

    def lazy_load(self) -> Iterator[Document]:
        for doc in self.manifest.documents:
            text = (self.text_dir / doc.file).read_text(encoding="utf-8").strip()
            yield Document(page_content=text, metadata=doc.model_dump())


def get_corpus_loader(config: CorpusConfig) -> CorpusLoader:
    """Создание сконфигурированного загрузчика документов."""
    return CorpusLoader(config=config)
