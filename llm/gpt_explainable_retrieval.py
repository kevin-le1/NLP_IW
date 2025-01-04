from __init__ import llm
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from prompt import PHISHING_PROMPT

class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = llm

        # Create a base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create an explanation chain
        explain_prompt = ChatPromptTemplate(
            messages=[
                "system: You are an AI assistant that explains how the context relates to a user's query.",
                f"user: {PHISHING_PROMPT}\n\n"
                "Query: {query}\n\n"
                "Context: {context}\n\n"
                "Explanation:"
            ],
            input_variables=["query", "context"],
        )
        
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        explained_results = []
        
        for doc in docs:
            # Generate explanation
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
        
        return explained_results