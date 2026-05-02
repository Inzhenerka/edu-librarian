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
    name: str
    description: str
    documents: list[CorpusDocument]

    @classmethod
    def from_yaml_file(cls, manifest_path: str | Path) -> Self:
        text = Path(manifest_path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.safe_load(text))
