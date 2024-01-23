# Repository for Generating Baselines for AI QoS

# How to Use?

## Main Controller
- generate_finalres.py

## PARTIES
### Program
- PARTIES.py
### Other Files Needed
- CONFIG: indicates the number of collocating models, names of each model and their corresponding target QoS

- LATENCY_FILE

- RUN_SCRIPT: script used to run the collocation experiment

- LOG_FILE

Paths to the above files need to be incidated in PARTIES.py

### Baselines
- results/data.data
- other files are the results for each individual run

## CLITE

### Program
- CLITE.py
### Other Files Needed
- LATENCY_FILE

- RUN_SCRIPT: script used to run the collocation experiment

- MAIN_LOG_FILE: used to record program output

- RUN_SCRIPT_LOG_FILE: Used to record changes made to RUN_SCRIPT by the program

Paths to the above files need to be incidated in LITE.py

### Baselines
- TBA