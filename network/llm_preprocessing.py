from langchain_ollama.llms import OllamaLLM
from scapy.all import rdpcap, Raw
from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.layers.dns import DNSQR, DNSRR
import os

# Function to extract HTTP and DNS data
def extract_data_from_packets(packets):
    extracted_data = []
    for packet in packets:
        # HTTP Request
        if packet.haslayer(HTTPRequest):
            http_layer = packet[HTTPRequest]
            extracted_data.append({
                "type": "HTTP Request",
                "method": http_layer.Method.decode() if http_layer.Method else "",
                "host": http_layer.Host.decode() if http_layer.Host else "",
                "path": http_layer.Path.decode() if http_layer.Path else "",
            })
            
        # HTTP Response
        elif packet.haslayer(HTTPResponse):
            http_layer = packet[HTTPResponse]
            extracted_data.append({
                "type": "HTTP Response",
                "status_code": http_layer.Status_Code.decode() if http_layer.Status_Code else "",
                "reason": http_layer.Reason_Phrase.decode() if http_layer.Reason_Phrase else "",
            })
            
        # DNS Query
        elif packet.haslayer(DNSQR):
            dns_layer = packet[DNSQR]
            extracted_data.append({
                "type": "DNS Query",
                "query_name": dns_layer.qname.decode() if dns_layer.qname else "",
            })
            
        # DNS Response
        elif packet.haslayer(DNSRR):
            dns_layer = packet[DNSRR]
            extracted_data.append({
                "type": "DNS Response",
                "response_name": dns_layer.rrname.decode() if dns_layer.rrname else "",
                "response_data": dns_layer.rdata.decode() if hasattr(dns_layer.rdata, "decode") else dns_layer.rdata,
            })
            
    return extracted_data

# Function to batch the data
def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
        
# Function to process each batch for an LLM
def prepare_batch_for_llm(batch):
    llm_input = ""
    for item in batch:
        if item["type"] == "HTTP Request":
            llm_input += f"HTTP Request - Method: {item['method']}, Host: {item['host']}, Path: {item['path']}\n"
        elif item["type"] == "HTTP Response":
            llm_input += f"HTTP Response - Status Code: {item['status_code']}, Reason: {item['reason']}\n"
        elif item["type"] == "DNS Query":
            llm_input += f"DNS Query - Query Name: {item['query_name']}\n"
        elif item["type"] == "DNS Response":
            llm_input += f"DNS Response - Response Name: {item['response_name']}, Response Data: {item['response_data']}\n"
    return llm_input

# Function to query LLM for each batch
def analyze_batch_with_llm(llm, batch_input):
    prompt = f"""
You are a cybersecurity expert. Analyze the following network packet data for any suspicious activities and identify any malicious elements. If you cannot detect any malicious elements then say "I cannot detect any malicious elements":

{batch_input}

Is there anything suspicious? If so, describe the malicious activity in detail.
"""
    response = llm.generate(prompts=[prompt])
    return response


def run(packets):
    # Define the number of batches (5)
    NUM_BATCHES = 5

    # Extract data from the PCAP file
    data = extract_data_from_packets(packets)

    # Calculate the number of packets in each batch
    total_packets = len(data)
    batch_size = total_packets // NUM_BATCHES  # Base size of each batch
    remainder = total_packets % NUM_BATCHES   # Remaining packets to distribute

    # Prepare the batches
    all_batches = []
    start_idx = 0

    for i in range(NUM_BATCHES):
        # Calculate the size of this batch (base size + 1 if remainder > 0)
        end_idx = start_idx + batch_size + (1 if i < remainder else 0)
        batch = data[start_idx:end_idx]
        all_batches.append(batch)

        # Update the starting index for the next batch
        start_idx = end_idx

    # Initialize LLM
    llm = OllamaLLM(model="llama3.1:latest", max_tokens=4000, temperature=0.2)

    # Analyze all batches
    results = []
    for i, batch in enumerate(all_batches):
        print(batch)
        print(f"Analyzing batch {i + 1}/{len(all_batches)}...")
        result = analyze_batch_with_llm(llm, batch)
        print(result)
        results.append({"batch": i + 1, "response": result})

    # Print results for each batch
    for result in results:
        print(f"Batch {result['batch']} Analysis:\n{result['response']}\n")

# Iterate over many PCAP data
def main():
    # directory = "/Users/kevin/Desktop/NLP_IW/network/ddos"
    
    # pcap_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pcap')]
    
    # # Load PCAP data
    # for file in pcap_files:
    #     packets = rdpcap(file)
    #     run(packets)
    
    run(rdpcap("/Users/kevin/Desktop/NLP_IW/network/hao123-com_packet-injection.pcap"))

# __name__
if __name__=="__main__":
    main()