from fastapi import FastAPI
from langserve import add_routes
from app import app

server = FastAPI(
    title="RAG Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(app=server, path="/chat", runnable=app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=server, host="localhost", port=8080)
