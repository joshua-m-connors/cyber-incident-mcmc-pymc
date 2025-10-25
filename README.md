Cyber Incident Estimation utilizing Markov Chain Monte Carlo (MCMC) simulation

This code takes an initial (prior) estimate of the frequency of attacks against an orgnaization. This range would be similar to Threat Event Frequency (TEF) in the Factor Analysis of Information Risk (FAIR) taxonomy.

It then simulates each attack as it progresses through the relevant MITRE ATT&CK tactics. Each tactic has an individually estimated range of control strength that gets applied.

The result is then an posterior projection of the number of successful attacks that will actually get through the entire attack process.

Sample Output:

<img width="1296" height="706" alt="Screenshot From 2025-10-25 15-49-40" src="https://github.com/user-attachments/assets/f3d1e7ae-06d7-489e-bd02-cf57800c9590" />

<img width="1400" height="1000" alt="Figure_1" src="https://github.com/user-attachments/assets/95eefe2c-0422-4b2e-b05c-61edb76e66d6" />

<img width="600" height="400" alt="Figure_2" src="https://github.com/user-attachments/assets/f5f979cd-e23e-49f2-81b4-2820d735e403" />

<img width="700" height="400" alt="Figure_3" src="https://github.com/user-attachments/assets/74f12427-9ee5-47d9-8817-d8d52fc167d0" />
