from rag.prompts import generate_prompt
from rag.config import llm_name, base_url
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(base_url=base_url, model=llm_name, temperature=0)

rag_chain = generate_prompt | llm | StrOutputParser()
