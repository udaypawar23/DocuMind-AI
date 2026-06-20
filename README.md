# 📄 DocuMind AI

AI-powered PDF Question Answering System using Retrieval-Augmented Generation (RAG).

## Features

* Upload PDF documents
* Semantic search using embeddings
* Question answering from PDF content
* FAISS vector database
* Ollama TinyLlama integration
* Gradio user interface

## Tech Stack

* Python
* LangChain
* FAISS
* Ollama
* TinyLlama
* Sentence Transformers
* Gradio

## Project Workflow

PDF Upload → Text Chunking → Embeddings → FAISS Vector Store → Retrieval → TinyLlama → Answer Generation

## How to Run

```bash
pip install -r requirements.txt
python gradio_app.py
```

## Future Enhancements

* PDF Summary Generation
* Multiple PDF Support
* Chat History
* Source Citations
* Cloud Deployment
