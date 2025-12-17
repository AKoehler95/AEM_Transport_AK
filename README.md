# AEM Transport
This repository provides a reactive contaminant transport model based on the Analytic Element Method. The AEM solution for steady-state plumes forming from complex contaminant sources is expressed as an infinite series of Mathieu functions. 
For detailed information please see Köhler et al. (2026) (PDF provided).

Models with multiple contaminant (donor) and accceptor sources can be set up easily via simple input files (simulation_config.json).
Additionally, the inverse model mode allows to automatically calibrate single model parameters for multiple sites simultaneously.

# Command for installing required libraries:

pip install -r requirements.txt