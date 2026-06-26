from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.llm import get_llm
from dotenv import load_dotenv

load_dotenv()

embeddings = get_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = get_llm()

print("\nRAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=5)

    # ---- DEBUG: see what retrieval actually returns ----
    print(f"\n[debug] retrieved {len(docs)} chunks")
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"[debug] context length: {len(context)} chars")
    print("[debug] first chunk:", repr(context[:200]), "\n")
    # ----------------------------------------------------

    prompt = f"""You are a helpful assistant answering questions about a document.
Answer ONLY using the context below. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    print("Answer:", response.content, "\n")
