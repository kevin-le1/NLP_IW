import ollama

message = "Is using PCAP data to find suspicious trends viable?"

stream = ollama.chat(
  model='llama3.1',
  messages=[{'role': 'user', 'content': f'{message}'}],
  stream=True
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)