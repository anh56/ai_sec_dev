import re
import csv
import sys

csv.field_size_limit(sys.maxsize)
import pandas as pd
from Levenshtein import ratio
import duckdb

con = duckdb.connect("./result/models.db")
df = con.execute("select * from models").df()

df["github_links_set"] = None
df["github_links_score"] = None
df["highest_score_link"] = None
df["highest_score"] = None

print(df.info())
print(len(df))

df.drop_duplicates(subset=["model_id"], inplace=True)
print(len(df))

df.dropna(subset=["github_links"], inplace=True)
print(len(df))

to_remove = [
    ")", "(", ",",
]

for index, row in df.iterrows():
    print(index, row["model_id"])
    link_str = row["github_links"]
    # df["github_links_set"] = df["github_links"].apply(extract_github_repo_link_set)
    # def extract_github_repo_link_set(link_str: str):
    if pd.isna(link_str) or link_str == "" or link_str == "[]":
        df.at[index, "github_links_set"] = None
        continue

    links = link_str.split(",")
    repo_link = None
    processed_links = set()
    # authors = set()
    # repos = set()

    for link in links:
        if link == "":
            continue
        link = link.strip()
        # remove special characters
        for char in to_remove:
            link = link.replace(char, "")

        # capture entire link
        if re.search(r"https?:\/\/?github\.com\/[\w-]+\/[\w-]+", link):
            repo_link = re.search(r"https?:\/\/?github\.com\/[\w-]+\/[\w-]+", link).group(0)
            processed_links.add(repo_link)
            author_repo = repo_link.replace("https://github.com/", "")

    # capture each group
    # author, repo = re.search(r"https?:\/\/?github\.com\/([\w-]+)\/([\w-]+)", link).groups()
    # authors.add(author)
    # repos.add(repo)

    if processed_links:
        processed_links = list(processed_links)
        df.at[index, "github_links_set"] = processed_links
        scores = [ratio(link, row["model_id"]) for link in processed_links]
        df.at[index, "github_links_score"] = scores
        df.at[index, "highest_score_link"] = processed_links[scores.index(max(scores))]
        df.at[index, "highest_score"] = max(scores)
    # else:
    # 	df.at[index, "github_links_set"] = None
    # 	df.at[index, "github_links_score"] = None

    print(list(processed_links))
# row["github_author"] = authors if authors else None
# row["github_repos"] = repos if repos else None

df.to_csv("./result/models_with_scored_link.csv", index=False)
