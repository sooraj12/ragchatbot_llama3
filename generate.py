from prompts import generate_prompt
from config import llm_name, base_url
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(base_url=base_url, model=llm_name, temperature=0)

rag_chain = generate_prompt | llm | StrOutputParser()


def generate_response(question, retriever):
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})

    print(generation)
