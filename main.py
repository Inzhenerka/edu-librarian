import os
import sys

from dotenv import load_dotenv
from loguru import logger

from edu_librarian.agent import Librarian

# Подгружаем OPENAI_API_KEY из .env (на платформе он подставляется из PR_LLM_KEY)
load_dotenv()

logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO").upper())

agent = Librarian(llm_key="api", debug=False)

PROMPTS = [
    "Кого старик Тарас приютил у себя на озере?",
    "Кто такой Никита Демидович?",
    "Какова природа Уральских гор?",
]

for prompt in PROMPTS:
    print(f"\n👤: {prompt}")
    response = agent.invoke(prompt=prompt, thread_id="1")
    print(f"🤖: {response.content}")

    if response.sources:
        print("📚 Источники:")
        for ref_num, meta in response.sources.items():
            print(f"   [{ref_num}] «{meta.title}», {meta.author} - {meta.chunk_id}")
    else:
        print("📚 Источники: (модель не процитировала ни одного)")
