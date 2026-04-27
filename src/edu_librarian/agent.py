import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from edu_librarian.chat_model import load_chat_model
from edu_librarian.config import Config
from edu_librarian.prompts import LibrarianPrompt


class LibAgentResponse(BaseModel):
    content: str
    thread_id: str


class LibAgent:
    _agent: CompiledStateGraph

    def __init__(self, llm_key: str, debug: bool = False):
        """Инициализация обертки агента."""

        # Загружаем конфигурацию
        config = Config.from_yaml_file("config.yml")
        llm_config = config.llms[llm_key]

        # Создаем модель чата
        chat_model = load_chat_model(llm_config=llm_config)

        # Создаем примитивного агента ReAct
        self._agent = create_agent(
            model=chat_model,
            tools=[],
            system_prompt=LibrarianPrompt().render_prompt(),
            checkpointer=InMemorySaver(),
            middleware=[
                ModelRetryMiddleware(max_retries=2, initial_delay=1),
                ModelCallLimitMiddleware(run_limit=4, exit_behavior="end"),
            ],
            debug=debug,
        )

    def invoke(self, prompt: str, thread_id: str | None = None) -> LibAgentResponse:
        """Вызываем агента и извлекаем ответ из массива сообщений"""

        # Ограничиваем длину запроса для экономии токенов
        prompt = prompt.strip()[:4000]

        # Используем принятый или создаем новый ID чата
        effective_thread_id = thread_id or str(uuid.uuid4())

        # Формируем и передаем агенту сообщение
        response = self._agent.invoke(
            input={"messages": HumanMessage(prompt)},
            config={"configurable": {"thread_id": effective_thread_id}},
        )

        # Формируем ответ из последнего сообщения и состояния
        return LibAgentResponse(
            content=response['messages'][-1].content,
            thread_id=effective_thread_id,
        )
