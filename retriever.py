from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from config import embedding_model_name, base_url


def setup_retriever(doc_splits):
    embedding_model = OllamaEmbeddings(base_url=base_url, model=embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents=doc_splits, collection_name="rag-chroma", embedding=embedding_model
    )
    retriever = vectorstore.as_retriever()

    return retriever
