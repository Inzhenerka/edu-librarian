from dotenv import load_dotenv

from edu_librarian.config import Config
from edu_librarian.rag.chunk import load_chunks
from edu_librarian.rag.corpus import get_corpus_loader
from edu_librarian.rag.embedder import get_embedder
from edu_librarian.rag.text_splitter import get_text_splitter

# Подгружаем OPENAI_API_KEY из .env (на платформе он подставляется из PR_LLM_KEY)
load_dotenv()

config = Config.from_yaml_file("config.yml")
loader = get_corpus_loader(config.rag.corpus)
splitter = get_text_splitter(config.rag.splitter)
embedder = get_embedder(config.rag.embedder)

# Берём все чанки уральского корпуса и эмбеддим одним батчем — для сотни чанков это один HTTP-вызов
chunks = load_chunks(loader, splitter)
texts = [chunk.page_content for chunk in chunks]
print(f"Эмбеддим {len(texts)} чанков корпуса через {config.rag.embedder.model}...")
vectors = embedder.embed_documents(texts)
print(f"Размерность вектора: {len(vectors[0])}\n")


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


# Считаем косинус для всех пар (i < j) — O(N^2) на сотне чанков это мгновенно
pairs = []
for i in range(len(chunks)):
    for j in range(i + 1, len(chunks)):
        pairs.append((i, j, cosine(vectors[i], vectors[j])))
pairs.sort(key=lambda p: -p[2])

print(f"Всего пар чанков: {len(pairs)}\n")


def show_pair(idx_i: int, idx_j: int, score: float) -> None:
    chunk_i = chunks[idx_i]
    chunk_j = chunks[idx_j]
    print(f"\n-> Соответствие: {score:.2f}")
    print(f"---------- Чанк {chunk_i.metadata['chunk_id']} ----------")
    print(f"«{chunk_i.page_content.replace('\n\n', '\n')}»")
    print(f"---------- Чанк {chunk_j.metadata['chunk_id']} ----------")
    print(f"«{chunk_j.page_content.replace('\n\n', '\n')}»")


print("Топ-5 близких пар:")
for i, j, score in pairs[:5]:
    show_pair(i, j, score)
