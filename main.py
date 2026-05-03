from edu_librarian.config import Config
from edu_librarian.rag.chunk import load_chunks, ChunkMetadata
from edu_librarian.rag.corpus import get_corpus_loader
from edu_librarian.rag.text_splitter import get_text_splitter


# 1. Поднимаем конфиг, лоадер корпуса и сплиттер
config = Config.from_yaml_file("config.yml")
loader = get_corpus_loader(config.rag.corpus)
splitter = get_text_splitter(config.rag.splitter)

# 2. Загружаем из корпуса чанки с метаданными и id
chunks = load_chunks(loader, splitter)

print(f"📚 Корпус: {loader.manifest.name}")
print(f"   Чанков всего: {len(chunks)}\n")

# 3. Первый чанк и его метаданные
chunk = chunks[0]
meta = ChunkMetadata.from_document(chunk)
print(f"📄 Первый чанк")
print(f"   id (UUID5):  {chunk.id}")
print(f"   chunk_id:    {meta.chunk_id}")
print(f"   start_index: {meta.start_index}")
print(f"   длина:       {len(chunk.page_content)} символов")
print(f"   текст:       «{chunk.page_content[:80].strip()}…»\n")
