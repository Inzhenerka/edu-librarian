from typing import NotRequired

from langchain.agents.middleware import AgentState
from langchain_core.documents import Document


class LibrarianAgentState(AgentState):
    chunks: NotRequired[list[Document]]
