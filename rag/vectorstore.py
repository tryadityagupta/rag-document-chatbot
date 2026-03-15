from langchain_community.vectorstores import FAISS


def build_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, path="vectorstore"):
    vectorstore.save_local(path)


def load_vectorstore(embeddings, path="vectorstore"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
