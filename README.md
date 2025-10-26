Cyber Incident Estimation utilizing Markov Chain Monte Carlo (MCMC) simulation

This code takes an initial (prior) estimate of the frequency of attacks against an orgnaization. This range would be similar to Threat Event Frequency (TEF) in the Factor Analysis of Information Risk (FAIR) taxonomy.

It then simulates each attack as it progresses through the relevant MITRE ATT&CK tactics. Each tactic has an individually estimated range of control strength that gets applied.

The result is then an posterior projection of the number of successful attacks that will actually get through the entire attack process. These are then combined with estimates of loss aligned to FAIR loss categories to compute Annual Loss.

Requires:
- Python
- PyMC

Sample Output:

<img width="1088" height="976" alt="Screenshot From 2025-10-25 21-35-23" src="https://github.com/user-attachments/assets/f9c6eb92-71af-4ae2-b91a-9351ffb360ad" />

<img width="1400" height="1000" alt="Figure_1" src="https://github.com/user-attachments/assets/53b82d52-3bba-45e6-a94a-1dfebef89c36" />

<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/62f49c11-e6d0-45f0-9bf9-661e77af6a38" />


Acknowledgements:
- Got lots of help on this from Anthropic Claude and ChatGPT.
