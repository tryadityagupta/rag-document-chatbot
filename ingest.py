from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import build_vectorstore, save_vectorstore

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/1.pdf"

print("Loading documents...")
docs = load_documents(PDF_PATH)

print("\nSplitting into chunks...")
chunks = split_documents(docs)

print("\nGenerating embeddings...")
embeddings = get_embeddings()

print("\nBuilding vectorstore...")
vectorstore = build_vectorstore(chunks, embeddings)

print("\nSaving vectorstore...")
save_vectorstore(vectorstore)

print("\nIngestion complete!")
