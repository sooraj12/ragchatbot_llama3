from loader import load_urls
from retriever import setup_retriever
from generate import generate_response
from graders import setup_retriever_grader, setup_hallucination_grader, setup_ans_grader
from router import setup_router

# load documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
doc_splits = load_urls(urls)

# setup retriever
retriever = setup_retriever(doc_splits)

## retrieval grader
retrieval_grader = setup_retriever_grader()

# hallucination grader
hallucination_grader = setup_hallucination_grader()

# answer grader
ans_grader = setup_ans_grader()

# question router
question_router = setup_router()
