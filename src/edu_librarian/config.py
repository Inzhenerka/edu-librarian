from typing import Literal, Self
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

type LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppConfig(BaseModel):
    log_level: LogLevelType = "INFO"


class LLMConfig(BaseModel):
    model: str
    provider: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_output_tokens: int = 512


class EmbeddingsConfig(BaseModel):
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    timeout: float = 30.0


class RAGConfig(BaseModel):
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)


class Config(BaseModel):
    app: AppConfig
    llms: dict[str, LLMConfig]
    rag: RAGConfig

    @classmethod
    def from_yaml_file(cls, config_path: str | Path = "config.yml") -> Self:
        config_str = Path(config_path).read_text(encoding="utf-8")
        config_dict = yaml.safe_load(config_str)
        return cls.model_validate(config_dict)
