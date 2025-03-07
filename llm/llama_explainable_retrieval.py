from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import seaborn as sns
import modal
from pydantic import PrivateAttr

modal_app = modal.App()

llama_function = modal.Function.lookup("example-tgi-Meta-Llama-3-70B-Instruct", "Model.generate")

class ModalLLMWrapper(LLM):
    _modal_function: Any = PrivateAttr()  # Declare as private attribute

    def __init__(self, modal_function):
        super().__init__()
        self._modal_function = modal_function  # Store the modal function

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        
        truncated_prompt = truncate_text(prompt, max_tokens=725)
    
        
        # Run modal function and get the response
        response = self._modal_function.remote(truncated_prompt)
        # Check if the response is a string or dictionary
        if isinstance(response, dict) and "text" in response:
            return response["text"]
        elif isinstance(response, str):
            return response
        else:
            raise ValueError("Unexpected response format from modal function")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"modal_function_identifier": "Meta-Llama-3-70B-Instruct"}
    
    @property
    def _llm_type(self) -> str:
        return "modal_llm"

# Define utility classes/functions
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OllamaEmbeddings()  # Use Ollama embeddings
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ModalLLMWrapper(llama_function)  # Wrap modal function for LangChain compatibility

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

        explain_prompt = ChatPromptTemplate(
            messages=[
                "system: You are an AI assistant that explains how the context relates to a user's query.",
                "user: Determine if the email content is phishing or not. Additionally, note that these emails have passed through our heuristics of testing (i.e. we have determined they aren't phishing emails). Explain why it is or is not considered phishing based on context, print at the end EXACTLY (not phishing) or (phishing).\n\n"
                "Query: {query}\n\n"
                "Context: {context}\n\n"
                "Explanation:"
            ],
            input_variables=["query", "context"],
        )

        # Now the explain chain uses the wrapped modal function
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        docs = self.retriever.invoke(query)
        explained_results = []

        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data)  # Use invoke to run the pipeline
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



def create_classification_report(Y_test, Y_pred):
    print('--------Classification Report---------\n')
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    metrices = [accuracy, f1, precision, recall, roc_auc]
    scores = pd.DataFrame(pd.Series(metrices).values, index=['accuracy', 'f1-score', 'precision', 'recall', 'roc auc score'], columns=['score'])
    print(scores)
    print('\n--------Plotting Confusion Matrix---------')
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, cmap='RdYlGn_r', annot_kws={'size': 16})
    return scores

def truncate_text(text, max_tokens):
    # this doesnt really work need to first tokenize the text modal is bad does not to it properly
    words = text.split()
    return ' '.join(words[:max_tokens])

if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("/Users/kevin/Desktop/NLP_IW/bert_nongpu/Phishing_Email.csv")

    # Assuming the dataset has columns 'Email Text' and 'Label' (1 for phishing, 0 for not phishing)
    X = data['Email Text']
    y = data['Email Type'].str.lower().str.strip().apply(lambda x: 1 if x == 'phishing email' else 0)

    # Split into train and test sets (80% train, 20% test)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.015, random_state=10)
    print(y_test.head(50))

    y_pred = []
    
    # Add this
    explainable_rag = ExplainableRAGMethod("")
    

    for idx, email_text in enumerate(X_test):
        # Initialize explainable RAG method with train texts
        # Test query
        query = f'Is the following email a phishing email? Note, if it is safe do not add any reasoning. : {email_text}'
        # Run the explainable RAG method
        results = explainable_rag.run(query)
        print(idx, results[0]['explanation'], '\n')
        if 'not phishing' in results[0]['explanation']:
            pred = 0  # Not phishing
        else:
            pred = 1  # Phishing
        y_pred.append(pred)
    

    # Generate the classification report
    create_classification_report(y_test, y_pred)
    
    print(y_test, y_pred)
    
    # Convert y_test and y_pred to a DataFrame
    results_df = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})

    # Save the DataFrame to a CSV file
    results_df.to_csv("/Users/kevin/Desktop/NLP_IW/llm/predictions.csv", index=False)

'''
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 1, 0]
'''