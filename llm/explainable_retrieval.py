from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
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
        docs = self.retriever.get_relevant_documents(query)
        explained_results = []

        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            print("Input Data:", input_data)
            explanation = self.explain_chain.invoke(input_data)
            print("Raw Explanation Output:", explanation)
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
def parse_args():
    parser = argparse.ArgumentParser(description="Explainable RAG Method")
    test = """Subject: Important Account Update Required!

            Dear Valued Customer,

            We noticed unusual activity in your account. To ensure your security, please verify your account information immediately by clicking the link below:

            [Verify Your Account](http://bankofamerica.com/verify)

            Failure to do so may result in account suspension.

            Thank you,
            Your Bank Security Team"""

    parser.add_argument('--query', type=str, default=f'Are the following emails {test} phishing emails', help="Query for the retriever")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Sample texts (these can be replaced by actual data)
    texts = [
        "The message contains a generic greeting and urges you to act quickly.",
        "Phishing emails often have links that lead to fake websites designed to steal your credentials.",
        "Legitimate emails from companies usually do not ask for sensitive information directly through email.",
        "Beware of attachments in unsolicited emails, as they may contain malware."
    ]

    explainable_rag = ExplainableRAGMethod(texts)
    results = explainable_rag.run(args.query)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Query: {result['query']}")
        print(f"Content: {result['content']}")
        print(f"Explanation: {result['explanation']}")
        print()


# Need to get dataset (1)

# (2) View accuracy between the two llama w/ content vs llama w/o content
