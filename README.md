Cyber Incident Estimation utilizing Markov Chain Monte Carlo (MCMC) simulation

This code takes an initial (prior) estimate of the frequency of attacks against an organization. This range would be similar to Threat Event Frequency (TEF) in the Factor Analysis of Information Risk (FAIR) taxonomy, but likely a bit higher (analysts likely weed out or are unaware of many attacks that fail early on).

Note: There is an option (lines 270 & 271) to enter actual observational data (i.e. if you know there were 2 successful incidents in the past 3 year).

It then simulates each attack as it progresses through the relevant MITRE ATT&CK tactics. Each tactic has an individually estimated range of control strength that gets applied. There is also logic that assumes that when an attacker fails with a tactic, they may retry and/or fallback and try a different path. However, as they fallback and try different techniques the chance of being discovered or blocked increases.

The result is then an posterior projection of the number of successful attacks that will actually get through the entire attack process to full compromise. These are then combined with estimates of loss aligned to FAIR loss categories to compute Annualized Loss Expectancy (ALE).

Running the code:

You can use either the .py file or if you prefer Jupyter notebooks you can use the .ipynb file.

Requires:
- Python
- PyMC (Recommended to install using Anaconda: https://www.pymc.io/projects/docs/en/latest/installation.html#)
- Jupyer Notebooks (optional)

Acknowledgements:
- I got a lot of help on this from Claude and ChatGPT.

Sample Output:

<img width="1084" height="490" alt="Screenshot From 2025-10-26 13-35-00" src="https://github.com/user-attachments/assets/821de248-64b1-4956-aa6e-ee9abea5e3b3" />

<img width="1500" height="1000" alt="Figure_1" src="https://github.com/user-attachments/assets/cc62637e-16df-4d4f-8a45-715281681cc1" />

<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/d70c75fe-79b2-49e6-9169-a303be566dc2" />
