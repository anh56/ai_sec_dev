import duckdb
import pandas as pd
from github import Github
from github import Auth
from concurrent.futures import ThreadPoolExecutor

auth = Auth.Token("")
g = Github(auth=auth)
conn = duckdb.connect('./result/models.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS gh_issues_comments (
        repo_name TEXT,
        issue_url TEXT,
        issue_title TEXT,
        issue_body TEXT,
        pr_from_issue TEXT,
        user_login TEXT,
        issue_number INTEGER,
        issue_comment TEXT,
    )
''')

issues = cursor.execute('SELECT * FROM gh_issues').fetchall()

def process_issue(issue):
    rate_limit = g.get_rate_limit()
    print(issue[1], issue[-1], rate_limit)
    try:
        issue_comments = g.get_repo(issue[1]).get_issue(issue[-1]).get_comments()
        for comment in issue_comments:
            print("Inserting ", issue[1], comment.id, rate_limit)
            cursor.execute(f'''
                INSERT INTO gh_issues_comments(
                repo_name,issue_url,issue_title,issue_body,pr_from_issue,user_login, issue_number, issue_comment
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (issue[1], issue[2], issue[3], issue[4], issue[5], issue[6], issue[7], comment.body))
    except Exception as e:
        print(f"ERROR {e}")

# def process_issue(issue):
#     issue_cursor = conn.cursor()
#     print(issue[0],issue[-1])
#     issue_comments = g.get_repo(issue[0]).get_issues_comments()
#     for comment in issue_comments:
#         issue_cursor.execute(f'''
#             INSERT INTO gh_issues_comments
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (issue[0], issue[1], issue[2], issue[3], issue[4], issue[5], issue[6], comment.body))

with ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(process_issue, issues)
