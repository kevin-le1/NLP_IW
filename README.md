# Domain-Specific Cybersecurity LLM

Accurate detection and classification of cybersecurity threats are critical for pro-
tecting sensitive data and infrastructure. This research proposes the development of
Retrieval-Augmented Generation (RAG) techniques with Large Language models
(LLMs), leveraging historical datasets of phishing emails and network traffic. This
system employs a classifier and analyzer to enhance threat identification, focusing
on phishing attacks and abnormal network behaviors. To test the system, we evalu-
ate the model on baselines such as the BERT model, heuristics, and base LLM on
various benchmarks, aiming to demonstrate that combining retrieval mechanisms
with LLMs improves domain-specific threat detection and analysis, attempting to
achieve high precision, recall, F1 scores, and qualitative analysis. Overall, this re-
search contributes to the improvement of cybersecurity by providing a new method
for creating an intelligent, dynamic, and responsive system capable of mitigating
and detecting evolving threats.

# bert
Contains Modal Lab training of Bert model

# bert_nongpu
Contains CPU training of Bert Model / Visualization of best model

# helper 
Contains helper functions for evaluating RAG

# llm
- Contains the bulk functionality of GPT explainable RAG
- Need to switch prompt to switch modes

# network
- Contains network packets of DDoS Attacks & Packet Injection Attacks
- Contains code for heuristic approach & llm approach

# apis
Contains beginning of IW work not important, but has some contents for potential future work