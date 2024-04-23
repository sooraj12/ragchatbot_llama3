from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from pprint import pprint

llm_host = "103.152.157.130:11434"
llm = "llama3:70b"
embedding_model_name = "nomic-embed-text"

## indexing
# load documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
# add to vector db
embedding_model = OllamaEmbeddings(
    base_url=f"http://{llm_host}", model=embedding_model_name
)
vectorstore = Chroma.from_documents(
    documents=doc_splits, collection_name="rag-chroma", embedding=embedding_model
)
retriever = vectorstore.as_retriever()
print(retriever.invoke("what is an agent"))

## retrieval grader
