from pathlib import Path

from edu_librarian.rag.splitter import iter_corpus_chunks


chunks = list(iter_corpus_chunks(Path("corpus")))

print(f"\nКорпус разрезан на {len(chunks)} чанков:\n")
for chunk in chunks[:4]:
    print(f"📄 {chunk.id}")
    print(f"   позиция [{chunk.start_pos}-{chunk.end_pos}]")
    print(f"   «{chunk.chunk_text[:100].strip()}...»")
    print(f"   metadata: {sorted(chunk.metadata.keys())}\n")

if len(chunks) > 4:
    print(f"...и ещё {len(chunks) - 4} чанков")
