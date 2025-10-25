Cyber Incident Estimation utilizing Markov Chain Monte Carlo (MCMC) simulation

This code takes an initial (prior) estimate of the frequency of attacks against an orgnaization. This range would be similar to Threat Event Frequency (TEF) in the Factor Analysis of Information Risk (FAIR) taxonomy.

It then simulates each attack as it progresses through the relevant MITRE ATT&CK tactics. Each tactic has an individually estimated range of control strength that gets applied.

The result is then an posterior projection of the number of successful attacks that will actually get through the entire attack process.

Sample Output:
<img width="1800" height="1335" alt="pyMC_charts" src="https://github.com/user-attachments/assets/2a918ebf-e7e7-48dd-8f3f-3224a91a1148" />

<img width="1130" height="884" alt="Screenshot From 2025-10-25 14-18-55" src="https://github.com/user-attachments/assets/1d929563-d60f-40b6-b3a7-99f7175a9582" />

<img width="1228" height="1006" alt="Screenshot From 2025-10-25 14-19-26" src="https://github.com/user-attachments/assets/bc9c42ea-fc7a-46d6-91a6-282795a0b679" />
