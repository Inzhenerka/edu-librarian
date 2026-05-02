from pathlib import Path

from edu_librarian.config import Config
from edu_librarian.rag.corpus import CorpusManifest

config = Config.from_yaml_file("config.yml")

# 1. Читаем манифест корпуса из файла
corpus_manifest = CorpusManifest.from_yaml_file(config.rag.corpus.manifest_file)

# 2. Извлекаем паспорт первого документа
doc = corpus_manifest.documents[0]

# 3. Читаем .txt-файл документа и берём начало текста
text = (config.rag.corpus.text_dir / doc.file).read_text(encoding="utf-8")
fragment = text[:100]

# 5. Печатаем паспорт документа и поля собранного чанка
print(f"📚 Документ: {doc.title}")
print(f"   Автор: {doc.author}")
print(f"   Раздел: {doc.section}")
print(f"   Источник: {doc.source} ({doc.source_url})")
print(f"   Фрагмент: {fragment}...")
