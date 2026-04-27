from typing import ClassVar
from pathlib import Path

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel


class BasePrompt(BaseModel):
    __file_path__: ClassVar[str | Path]

    def render_prompt(self) -> str:
        text = Path(self.__file_path__).read_text(encoding="utf-8")
        template = Template(text, undefined=StrictUndefined)
        return template.render(self.model_dump()).strip()


class LibrarianPrompt(BasePrompt):
    __file_path__ = "prompts/templates/librarian.jinja"
