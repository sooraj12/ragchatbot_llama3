from langserve import RemoteRunnable
from pprint import pprint

remote_chain = RemoteRunnable("http://localhost:8080/chat/")

inputs = {"question": "What are the types of agent memory?"}
# inputs = {"question": "What is langchain?"}
# inputs = {"question": "What is ReactJS?"}

for output in remote_chain.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
