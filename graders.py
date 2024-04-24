from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from prompts import (
    retriever_grader_prompt,
    hallucination_grader_prompt,
    ans_grader_prompt,
)
from config import grader_llm_name, base_url


grader_llm = ChatOllama(
    base_url=base_url, model=grader_llm_name, format="json", temperature=0
)


def setup_retriever_grader():
    retrieval_grader = retriever_grader_prompt | grader_llm | JsonOutputParser()
    return retrieval_grader


def setup_hallucination_grader():
    hallucination_grader = hallucination_grader_prompt | grader_llm | JsonOutputParser()
    return hallucination_grader


def setup_ans_grader():
    ans_grader = ans_grader_prompt | grader_llm | JsonOutputParser()
    return ans_grader
