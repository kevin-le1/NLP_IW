# Instantiate an LLM instance & initialize it for future use.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# CHANGE BETWEEN 'ft:gpt-4o-mini-2024-07-18:mahsa-bastankhah:blockchain-tuned:AcRAw66O' and 'gpt-4o-mini'
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000)