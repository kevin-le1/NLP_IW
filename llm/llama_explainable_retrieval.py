from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import argparse

# Define utility classes/functions
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OllamaEmbeddings()  # Use Ollama embeddings
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = OllamaLLM(model="llama3.1:latest", max_tokens=4000, temperature=0.2)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})


        # Update the ChatPromptTemplate initialization
        explain_prompt = ChatPromptTemplate(
            messages=[
                "system: You are an AI assistant that explains how the context relates to a user's query.",
                "user: Determine if the email content is phishing or not. Additionally, note that these emails have passed through our heuristics of testing (i.e. we have determined they aren't phishing emails). Explain why it is or is not considered phishing based on context, printing at the end 0 (not phishing) or 1 (phishing).\n\n"
                "Query: {query}\n\n"
                "Context: {context}\n\n"
                "Explanation:"
            ],
            input_variables=["query", "context"],
        )

        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        docs = self.retriever.invoke(query)
        explained_results = []

        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data)
            explained_results.append({
                "query": query,
                "content": doc.page_content,
                "explanation": explanation
            })
        return explained_results


class ExplainableRAGMethod:
    def __init__(self, texts):
        self.explainable_retriever = ExplainableRetriever(texts)

    def run(self, query):
        return self.explainable_retriever.retrieve_and_explain(query)


# Argument Parsing
def parse_args(text):
    parser = argparse.ArgumentParser(description="Explainable RAG Method")
    test = f"""{text}"""

    parser.add_argument('--query', type=str, default=f'Is the following email a phishing email? Note, if it is safe do not add any reasoning. : {test}', help="Query for the retriever")
    return parser.parse_args()

if __name__ == "__main__":
    
    test = pd.read_csv("/Users/kevin/Desktop/NLP_IW/bert_nongpu/Phishing_Email.csv")
    args = parse_args(test["Email Text"][0])

    # Create 2 texts oriented from dataset
    texts = [
        "No additional information."
    ]

    explainable_rag = ExplainableRAGMethod(texts)
    results = explainable_rag.run(args.query)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Query: {result['query']}")
        print(f"Content: {result['content']}")
        print(f"Explanation: {result['explanation']}")
        print()
    
    if "(not phishing)" in results[0]['explanation']:
        print("hit")


# Need to get dataset (1)

# (2) View accuracy between the two llama w/ content vs llama w/o content
