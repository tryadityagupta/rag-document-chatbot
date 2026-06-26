from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.llm import get_llm

load_dotenv()

# Loaded once at startup, reused for every request (no reloading per call).
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading RAG pipeline...")
    embeddings = get_embeddings()
    state["vectorstore"] = load_vectorstore(embeddings)
    state["llm"] = get_llm()
    print("Ready.")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    question: str


PROMPT = """You are a helpful assistant answering questions about a document.
Answer ONLY using the context below. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""


@app.post("/chat")
def chat(q: Query):
    question = q.question.strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    docs = state["vectorstore"].similarity_search(question, k=5)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = PROMPT.format(context=context, question=question)
    answer = state["llm"].invoke(prompt).content

    sources = [d.page_content.strip() for d in docs]
    return {"answer": answer, "sources": sources}


@app.get("/")
def index():
    return FileResponse("static/index.html")
