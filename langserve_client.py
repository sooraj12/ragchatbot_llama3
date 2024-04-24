from langserve import RemoteRunnable
from pprint import pprint
import sys

args = sys.argv[1:]

if len(args) != 1:
    print("Usage: Provide question")
    sys.exit(1)

question = args[0]
print(question)

remote_chain = RemoteRunnable("http://localhost:8080/chat/")

inputs = {"question": question}

for output in remote_chain.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
