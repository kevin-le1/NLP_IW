from __init__ import llm
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant trained on blockchain concepts."),
    HumanMessage(content="Can you explain how a smart contract works?")
] 

response = llm(messages)

print(response)

print(response.content)

