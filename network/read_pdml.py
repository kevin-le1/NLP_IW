from scapy.all import rdpcap, Raw
from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.layers.dns import DNSQR, DNSRR

# Load PCAP
packets = rdpcap("/Users/kevin/Desktop/NLP_IW/network/hao123-com_packet-injection.pcap")

# Process Packets
for packet in packets:
    # Check for HTTP Requests
    if packet.haslayer(HTTPRequest):
        host = packet["HTTPRequest"].Host.decode(errors='ignore')
        path = packet["HTTPRequest"].Path.decode(errors='ignore')
        print(f"HTTP Request -> Host: {host}, Path: {path}")

    # Check for HTTP Responses
    if packet.haslayer(Raw):
        try:
            payload = packet[Raw].load.decode(errors='ignore')
            if "hao123.com" in payload or "02995.com" in payload:
                print(f"HTTP Response Payload: {payload}")
        except Exception as e:
            pass  # Ignore decoding errors

    # Check for DNS Queries
    if packet.haslayer(DNSQR):
        
        query = packet["DNSQR"].qname.decode(errors='ignore')
        print(f"DNS Query -> {query}")

    # Check for DNS Answers
    if packet.haslayer(DNSRR):
        for i in range(packet["DNS"].ancount):
            answer = packet["DNS"].an[i].rdata
            print(f"DNS Answer -> {answer}")

# Add correlation for redirection evidence
print("\n--- Analysis Summary ---")
for packet in packets:
    if packet.haslayer(HTTPRequest) and "02995.com" in packet["HTTPRequest"].Host.decode(errors='ignore'):
        print(f"Found HTTP Request to www.02995.com: {packet.summary()}")
    if packet.haslayer(Raw) and "hao123.com" in packet[Raw].load.decode(errors='ignore'):
        print(f"Potential Redirect to www.hao123.com: {packet.summary()}")
