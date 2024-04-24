from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8080/chat/")
print(remote_chain.invoke({"question": "What are the types of agent memory?"}))
