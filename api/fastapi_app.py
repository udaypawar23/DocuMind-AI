from fastapi import FastAPI
from pydantic import BaseModel

from src.embeddings import get_embedding_model
from src.retriever import load_vector_store
from src.rag_chain import get_llm

app = FastAPI(title="DocuMind AI API")

embedding_model = get_embedding_model()
retriever = load_vector_store(embedding_model)
llm = get_llm()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "DocuMind AI API Running"
    }

@app.post("/ask")
def ask(query: Query):

    docs = retriever.invoke(query.question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only using the provided context.

Context:
{context}

Question:
{query.question}
"""

    answer = llm.invoke(prompt)

    return {
        "question": query.question,
        "answer": answer
    }