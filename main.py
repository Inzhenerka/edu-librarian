from fastapi.testclient import TestClient
from edu_librarian.api import app, LibrarianResponse

PROMPTS = [
    "Кто такие Демидовы?",
    "Как их всех звали?",
]

with TestClient(app) as client:
    for prompt in PROMPTS:
        print(f"\n👤: {prompt}")
        api_response = client.post("/ask", data={"question": prompt, "thread_id": "1"})
        response = LibrarianResponse.model_validate(api_response.json())
        print(f"🤖: {response.content}")
        if response.sources:
            print("📚 Источники:")
            for ref_num, meta in response.sources.items():
                print(f"   [{ref_num}] «{meta.title}», {meta.author} - {meta.chunk_id}")
        else:
            print("📚 Источники: (модель не процитировала ни одного)")
