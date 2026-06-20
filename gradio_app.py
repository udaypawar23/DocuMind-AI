import gradio as gr

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import load_vector_store
from src.rag_chain import get_llm

# Load embedding model once
embedding_model = get_embedding_model()

# Global retriever
retriever = None


def process_pdf(pdf_file):
    global retriever

    try:
        # Load PDF
        docs = load_pdf(pdf_file.name)

        # Split into chunks
        chunks = split_documents(docs)

        # Create FAISS vector store
        create_vector_store(
            chunks,
            embedding_model
        )

        # Load retriever
        retriever = load_vector_store(
            embedding_model
        )

        return "✅ PDF processed successfully! You can now ask questions."

    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_document(question):
    global retriever

    try:
        if retriever is None:
            return "❌ Please upload and process a PDF first."

        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Load TinyLlama
        llm = get_llm()

        prompt = f"""
You are a helpful document assistant.

Use ONLY the provided context to answer the question.

If the user asks about the whole document,
provide a short summary.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        return response

    except Exception as e:
        return f"❌ Error: {str(e)}"


with gr.Blocks(title="DocuMind AI") as demo:

    gr.Markdown(
        """
# 📄 DocuMind AI

### Enterprise Document Assistant

Upload a PDF, process it, and ask questions about its content.
"""
    )

    pdf_file = gr.File(
        label="📂 Upload PDF",
        file_types=[".pdf"]
    )

    process_btn = gr.Button("⚙️ Process PDF")

    status = gr.Textbox(
        label="Status",
        value="Waiting for PDF upload..."
    )

    question = gr.Textbox(
        label="❓ Ask a Question",
        placeholder="What is this PDF about?"
    )

    submit_btn = gr.Button("🚀 Submit")

    answer = gr.Textbox(
        label="💡 Answer",
        lines=12
    )

    process_btn.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )

    submit_btn.click(
        fn=ask_document,
        inputs=question,
        outputs=answer
    )

demo.launch()