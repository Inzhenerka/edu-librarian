from dotenv import load_dotenv

from edu_librarian.agent import Librarian

load_dotenv()

agent = Librarian(llm_key="api")

QUESTIONS = [
    "Какие заводы получил Никита Демидов в 1702 году?",
    "Чем богат Урал по описанию Н. И. Березина в его очерке 1910 года?",
]

thread_id = None

for q in QUESTIONS:
    print(f"\n\n❓ {q}\n")
    answer = agent.invoke(q, thread_id=thread_id)
    thread_id = answer.thread_id
    print(f"📜 {answer.content}")
