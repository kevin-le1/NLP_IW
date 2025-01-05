PHISHING_PROMPT = '''
Determine if the email content is phishing or not. 
Explain why it is or is not considered phishing based on context, print at 
the end EXACTLY (not phishing) or (phishing).\n\n"
'''

DDOS_PROMPT = '''
Determine if the network traffic pattern indicates a Distributed Denial of Service (DDoS) attack or not.
Explain why it is or is not considered a DDoS attack based on the observed behavior and context. 
Print at the end EXACTLY (not DDoS) or (DDoS).\n\n"
'''

PACKET_ATTACK_PROMPT = '''
Determine if the captured packet data indicates a packet-based attack or not. 
Explain why it is or is not considered a packet-based attack based on the packet content and context. 
Print at the end EXACTLY (not packet attack) or (packet attack).\n\n"
'''