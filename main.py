from dotenv import load_dotenv

from edu_librarian.config import Config
from edu_librarian.rag.chunk import ChunkMetadata
from edu_librarian.rag.ingestion import ingest_corpus
from edu_librarian.rag.retriever import get_retriever

# Подгружаем OPENAI_API_KEY из .env (на платформе он подставляется из PR_LLM_KEY)
load_dotenv()

# Загружаем конфиг
config = Config.from_yaml_file("config.yml")

# Подключаемся к базе и загружаем в нее корпус
vector_store = ingest_corpus(config=config.rag, force=False)

# Получаем настроенный ретривер из базы
retriever = get_retriever(config=config.rag.retriever, vector_store=vector_store)

QUERIES = [
    "Кого старик Тарас приютил у себя на озере?",
    "Никита Демидов был уральским лебедем-приемышем",
    "Каково население Екатеринбурга?",
    "Как обучить свою LLM?",
    "London is the capital of Great Britain",
]

for query in QUERIES:
    print(f"\n❓ {query}")
    # Получаем из базы топ чанков с помощью ретривера
    chunks = retriever.invoke(query)
    print(f"📄 Найдено чанков: {len(chunks)}. ID: {[ChunkMetadata.from_document(c).chunk_id for c in chunks]}")
