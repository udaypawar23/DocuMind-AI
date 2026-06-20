import streamlit as st

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import load_vector_store
from src.rag_chain import get_llm

st.title("DocuMind AI")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_pdf("temp.pdf")

    chunks = split_documents(docs)

    embedding_model = get_embedding_model()

    create_vector_store(
        chunks,
        embedding_model
    )

    retriever = load_vector_store(
        embedding_model
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        retrieved_docs = retriever.invoke(
            question
        )

        context = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        llm = get_llm()

        prompt = f"""
        Answer only from the context.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)