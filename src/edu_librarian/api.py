from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from edu_librarian.agent import LibrarianResponse
from edu_librarian.dependencies import AgentDependency, init_global_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_global_dependencies(app)
    logger.info("Starting librarian")
    yield
    logger.info("Stopping librarian")


app = FastAPI(lifespan=lifespan)


@app.get("/demo")
def demo():
    """Simple demo of web UI."""
    return FileResponse("templates/demo.html")


@app.post("/ask", response_model=LibrarianResponse)
def ask(
    agent: AgentDependency,
    question: str = Form(
        description="Student question",
        examples=["Кого старик Тарас приютил у себя на озере?"],
    ),
    thread_id: str | None = Form(default=None, description="Chat thread id"),
) -> LibrarianResponse:
    """Ask question to librarian RAG agent."""
    try:
        return agent.invoke(prompt=question, thread_id=thread_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
