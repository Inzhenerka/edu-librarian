from typing import Annotated
import sys

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from loguru import logger

from edu_librarian.agent import Librarian
from edu_librarian.config import Config
from edu_librarian.rag.ingestion import ingest_corpus


def init_global_dependencies(app: FastAPI) -> None:
    """Инициализация и подготовка всех зависимостей агента."""
    # Загрузка переменных из .env
    load_dotenv()

    # Загрузка конфигурации
    config = Config.from_yaml_file("config.yml")

    # Настройка глобального логгера из конфига
    logger.remove()
    logger.add(sys.stdout, level=config.app.log_level)

    # Подготовка векторного хранилища и инджестинг корпуса
    vector_store = ingest_corpus(config.rag)
    logger.info(f"Vector store ready: {config.rag.store.collection}")

    # Создание агента
    app.state.agent = Librarian(
        llm_key="api",
        config=config,
        vector_store=vector_store,
        debug=False,
    )


def get_agent(request: Request) -> Librarian:
    return request.app.state.agent


type AgentDependency = Annotated[Librarian, Depends(get_agent)]
