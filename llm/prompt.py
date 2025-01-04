PHISHING_PROMPT = '''
Determine if the email content is phishing or not. 
Additionally, note that these emails have passed through 
our heuristics of testing (i.e. we have determined they aren't phishing emails).
Explain why it is or is not considered phishing based on context, print at 
the end EXACTLY (not phishing) or (phishing).\n\n"
'''

DDOS_PROMPT = '''
Determine if the network traffic pattern indicates a Distributed Denial of Service (DDoS) attack or not. 
Additionally, note that these patterns have passed through our initial traffic anomaly detection (i.e., we have determined they may warrant further investigation).
Explain why it is or is not considered a DDoS attack based on the observed behavior and context. 
Print at the end EXACTLY (not DDoS) or (DDoS).\n\n"
'''

PACKET_ATTACK_PROMPT = '''
Determine if the captured packet data indicates a packet-based attack or not. 
Additionally, note that these packets have passed through our initial intrusion detection system (i.e., we have flagged them as potentially malicious).
Explain why it is or is not considered a packet-based attack based on the packet content and context. 
Print at the end EXACTLY (not packet attack) or (packet attack).\n\n"
'''