from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.llm import get_llm

print("\nLoading embeddings...")
embeddings = get_embeddings()

print("\nLoading vectorstore...")
vectorstore = load_vectorstore(embeddings)

print("\nLoading LLM...")
llm = get_llm()

print("\nRAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=5)

    docs = docs[:3]

    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    docs = unique_docs

    context = "\n\n".join([doc.page_content for doc in docs])

    print("\nRetrieved Context:\n")

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:")
        print(doc.page_content)
        print("-" * 40)

    prompt = f"""

You are a helpful assistant answering questions about a document.

Rules:
1. Answer ONLY using the provided context.
2. Do not invent information.
3. If the answer is not in the context, say "I don't know".

Context: 
{context}

Question: {query}

Answer:
"""

    response = llm(prompt, max_new_tokens=80, max_length=None)

    answer = response[0]["generated_text"]
    answer = answer.split("Answer:")[-1]
    answer = answer.split("Question:")[0]
    answer = answer.strip()

    print("\nAnswer: ", answer, "\n")
