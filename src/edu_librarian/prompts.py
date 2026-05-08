from typing import ClassVar
from pathlib import Path

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain.messages import HumanMessage


class BasePrompt(BaseModel):
    __file_path__: ClassVar[str | Path]

    def render_prompt(self) -> str:
        text = Path(self.__file_path__).read_text(encoding="utf-8")
        template = Template(text, undefined=StrictUndefined)
        return template.render(self.model_dump()).strip()

    def render_human_message(self) -> HumanMessage:
        return HumanMessage(content=self.render_prompt())


class LibrarianPrompt(BasePrompt):
    __file_path__ = "prompts/templates/librarian.jinja"


class LibrarianHumanMessage(BasePrompt):
    __file_path__ = "prompts/messages/librarian_human.jinja"
    question: str
    chunks: list[Document]
