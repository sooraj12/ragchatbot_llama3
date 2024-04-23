from prompts import router_prompt
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from config import llm_name, base_url

router_llm = ChatOllama(base_url=base_url, model=llm_name, format="json", temperature=0)


def setup_question_router():
    question_router = router_prompt | router_llm | JsonOutputParser()
    return question_router
