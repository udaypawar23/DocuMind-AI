from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import load_vector_store
from src.rag_chain import get_llm

pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

embedding_model = get_embedding_model()

create_vector_store(
    chunks,
    embedding_model
)

retriever = load_vector_store(
    embedding_model
)

query = input("Ask a question: ")

retrieved_docs = retriever.invoke(query)

context = "\n".join(
    [doc.page_content for doc in retrieved_docs]
)

llm = get_llm()

prompt = f"""
Answer only from the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)