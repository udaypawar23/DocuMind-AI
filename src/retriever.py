from langchain_community.vectorstores import FAISS

def load_vector_store(embedding_model):

    vector_store = FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store.as_retriever()