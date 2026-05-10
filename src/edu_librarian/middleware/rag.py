from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.runtime import Runtime
from loguru import logger

from edu_librarian.state import LibrarianAgentState
from edu_librarian.prompts import LibrarianHumanMessage


class RAGMiddleware(AgentMiddleware[LibrarianAgentState]):
    """Готовая RAG-обвязка для агента. Нужно лишь подключить ее, передав ретривер."""

    state_schema = LibrarianAgentState

    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self._retriever = retriever

    def before_agent(
        self,
        state: LibrarianAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Готовим чанки под текущую тему беседы."""

        # Создаем запрос для ретривера из свежей истории сообщений
        query = build_retrieval_query(state, max_messages=3)

        # Получаем чанки с помощью ретривера
        chunks = self._retriever.invoke(query)
        logger.debug(f"RAG retrieved {len(chunks)} chunks")

        # Обновляем состояние
        return {"chunks": chunks}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Подмешиваем чанки в последнее сообщение пользователя, не оставляя их в состоянии."""

        # Извлекаем подготовленные чанки из состояния
        chunks = request.state.get("chunks") or []
        # Извлекаем последнее сообщения для переработки
        last_message = request.messages[-1]

        # Рендерим новое сообщение пользователя
        enriched_message = LibrarianHumanMessage(
            chunks=chunks,
            question=str(last_message.content),
        ).render_human_message()

        # Заменяем последнее сообщение для отправки в LLM
        new_messages = [*request.messages[:-1], enriched_message]
        logger.debug(f"User message enriched with chunks:\n{enriched_message.content}")
        return handler(request.override(messages=new_messages))


def build_retrieval_query(state: LibrarianAgentState, max_messages: int = 3) -> str:
    """Собрать запрос для ретривера из последних N сообщений пользователя, чтобы не терять контекст."""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return "\n".join(str(m.content) for m in human_messages[-max_messages:])
