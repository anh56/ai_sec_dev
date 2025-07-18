# Securing the AI Supply Chain: What Can We Learn From Developer-Reported Security Issues and Solutions of AI Projects?

This repository contains the data and scripts used in the paper
"Securing the AI Supply Chain: What Can We Learn From
Developer-Reported Security Issues and Solutions of AI Projects?".

# Main Structure
```
.
|-- mining/                     # Data collection scripts
|   |-- result/                     # Data collection results
|-- classifier/                 # Classifiers
|   |-- manual/                     # Manual label train-val
|   |-- test/                       # Manual label test
|   |-- process.ipynb               # Data processing from mining
|   |-- dl_tune.py                  # DL models
|   |-- llm.py                      # LLM zeroshot
|   |-- llm_few.py                  # LLM fewshot
|   |-- analysis.ipynb              # quantitative results (RQ1)
|-- theme/                      # Thematic analysis data
|   |-- data.csv                    # Raw data with sources, repo id, and discussion/issue number
|   |-- keypoints_codes.csv         # Keypoint extraction and codes identification
|   |-- codebook_debriefing.xlsx    # Codes and theme synthesis + peer debriefing process
|   |------------------------
|   |-- mapped.csv                  # These files are extracted versions
|   |-- issue_theme_summary.csv     # of the thematic codes to generate
|   |-- solution_theme_summary.csv  # quantitative results 
|   |-- analysis.ipynb              # for RQ2 and RQ3
```

# Requirements
Check the `requirements.txt` file for the necessary Python packages.

# Data
Including the full data with the discussion is too heavy so we only include the id and the sources along with the label. Full discussion data can be found the models.db database.

models.db: duckdb database containing the discussion data, contains the following tables:
- models: model metadata (model_id, downloads, downloads_all_time, likes, trending_score, pipeline_tags, tags, card_data, base_model_from_card_data, scan_done, files_with_issues, adapter_count, merge_count, quantized_count, finetune_count, github_links, kw_in_hf_readme)
- hf_discussions: discussions of models on HF (model_id, num, title, git_ref, url, keywords)
- hf_discussion_events: comments of discussions on HF (model_id, num, event_id, event_type, content, keywords)
- gh_repositories: GitHub repositories metadata (id, github_link, repo_name)
- gh_discussions: discussions of GitHub repositories (repo_name, discussion_number,discussion_title,discussion_body,author_login)
- gh_comments: comments of GitHub discussions (repo_name, discussion_number,author_login,comment_body)
- gh_issues: issues of GitHub repositories (repo_name,issue_url,issue_title,issue_body,pr_from_issue,user_login,issue_number)
- gh_issues_comments: comments of issues on GitHub repositories (repo_name,issue_url,issue_title,issue_body,pr_from_issue,user_login, issue_number, issue_comment)
SQL can be used to review the content of these tables.

# Experiments (RQ1)
- Run [process.ipynb](classifier/process.ipynb) notebook to extract the data from the database.
- [sample.sh](classifier/sample.sh) contains a list of sample commands to run the classifiers in different settings.
- [inference/distilbert](classifier/inference/distilbert) contains the inferred results using distilbert.
- Run [analysis.ipynb](classifier/analysis.ipynb) to generate the results of RQ1.

# Thematic Analysis (RQ2,3)
- The samples presented in the paper are presented as {Repo Name}-{Discussion Number}. 
- Only the last number is the discussion number, for example: comfyanonymous/ComfyUI-5165 -> repo name: comfyanonymous/ComfyUI, discussion number: 5165.
- These repo names and discussion number then can be used to trace back the source it originated from in [data.xlsx](theme/data.xlsx).
- Run [analysis.ipynb](theme/analysis.ipynb) to generate the frequency/coverage.

# Citation
TBD

