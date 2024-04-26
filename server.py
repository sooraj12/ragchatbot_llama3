from fastapi import FastAPI
from langserve import add_routes
from rag.rag import rag
from langchain.pydantic_v1 import BaseModel

server = FastAPI(
    title="RAG Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


class Input(BaseModel):
    question: str


add_routes(app=server, path="/chat", runnable=rag.with_types(input_type=Input))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=server, host="localhost", port=8080)
