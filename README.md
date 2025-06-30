# Security of AI Open-Source Projects - A Study of Developer-Reported Issues and Solutions

This repository contains the data and scripts used in the paper
"Security of AI Open-Source Projects - A Study of Developer-Reported Issues and Solutions".

# Requirements
Check the `requirements.txt` file for the necessary Python packages.

# Structure

```
.
|-- mining/             # Code for data collection
|   |-- result/             # Data collection results
|-- classifier/         # Code for classifiers
|   |-- manual/             # Manual label train-val
|   |-- test/               # Manual label test
|   |-- process.ipynb       # Data processing
|   |-- dl_tune.py          # DL models
|   |-- llm.py              # LLM zeroshot
|   |-- llm_few.py          # LLM fewshot
|-- theme/              # Thematic analysis data


```

# Data
Including the full data with the discussion is too heavy so we only include the id and the sources along with the label. Full discussion data can be found the models.db database.


# Citation
TBD

