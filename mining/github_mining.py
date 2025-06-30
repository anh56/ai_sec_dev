import duckdb
import pandas as pd
from github import Github
from github import Auth

auth = Auth.Token("")
g = Github(auth=auth)


def mine_repos(con, disable_commits=True):
	df = pd.read_csv('result/hf_gh_link_set_full_counts_0902.csv')
	cursor = con.cursor()

	for index, row in df.iterrows():
		print("Current rate limit", g.get_rate_limit())
		print(index, row["highest_score_link"])
		repo_name = row["highest_score_link"].replace("https://github.com/", "")
		repo = None
		try:
			repo = g.get_repo(repo_name)
		except Exception as e:
			print(f"Error when getting repo: {e}")
			continue

		repo_db = cursor.execute('''
		    INSERT INTO gh_repositories (github_link, repo_name)
		    VALUES (?, ?)
		''', (row["highest_score_link"], repo_name))
		# repo_db_id = repo_db.fetchone()[0]

		# Get all commit messages
		if not disable_commits:
			commits = repo.get_commits()
			for commit in commits:
				print(commit.url)
				cursor.execute('''
					INSERT INTO gh_commits (repo_name, commit_url, commit_message)
					VALUES (?, ?, ?)
				''', (repo_name, commit.url, commit.commit.message))

		try:
			# https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions
			discussions = repo.get_discussions(
				discussion_graphql_schema="""
				id
				title
				bodyText
				number
				bodyText
				author {
				  login
				}
				"""
			)
			for discussion in discussions:
				print(f"{repo_name}, discussion {discussion.number},")
				discussion_db = cursor.execute('''
					INSERT INTO gh_discussions (repo_name, discussion_number, discussion_title, discussion_body, author_login)
					VALUES (?, ?, ?, ?,?)
				''', (repo_name, discussion.number, discussion.title, discussion.body_text, discussion.author.login))
				# discussion_db_id = discussion_db.fetchone()[0]
				comments = discussion.get_comments(
					comment_graphql_schema="""
					id
					bodyText
					author {
					  login
					}
					"""
				)
				for comment in comments:
					print(f"{repo_name}, discussion {discussion.number}, comment {comment.id}")
					cursor.execute('''
						INSERT INTO gh_comments (repo_name, discussion_number, author_login, comment_body)
						VALUES (?, ?, ?, ?)
					''', (repo_name, discussion.number, comment.author.login, comment.body_text))

		# discussion_titles = [discussion.title for discussion in discussions]
		# print("Discussions:", discussion_titles)
		except Exception as exc:
			print(f"ERROR {exc}")

		# Get all issues
		issues = repo.get_issues(state="all")
		for issue in issues:
			print(f"{repo_name}, {issue.url}")
			pr_from_issue = None
			if issue.pull_request:
				pr_from_issue = issue.pull_request.html_url
			cursor.execute('''
			    INSERT INTO gh_issues (repo_name,issue_url,issue_title,issue_body,pr_from_issue,user_login, issue_number)
			    VALUES (?, ?, ?, ?, ?, ?, ?)
			''', (repo_name, issue.url, issue.title, issue.body, pr_from_issue, issue.user.login, issue.number))


# issue_titles = [issue.title for issue in issues]
# print("Issues:", issue_titles)


def get_repo_db():
	conn = duckdb.connect('./result/models.db')
	cursor = conn.cursor()
	cursor.execute("""
	CREATE SEQUENCE IF NOT EXISTS id_repo START 1;
	CREATE SEQUENCE IF NOT EXISTS id_commit START 1;
	CREATE SEQUENCE IF NOT EXISTS id_discussion START 1;
	CREATE SEQUENCE IF NOT EXISTS id_comment START 1;
	CREATE SEQUENCE IF NOT EXISTS id_issues START 1;
	""")

	cursor.execute('''
	    CREATE TABLE IF NOT EXISTS gh_repositories (
	        id INTEGER PRIMARY KEY DEFAULT nextval('id_repo'),
	        github_link TEXT,
	        repo_name TEXT,
	    )
	''')

	cursor.execute('''
	    CREATE TABLE IF NOT EXISTS gh_commits (
	        id INTEGER PRIMARY KEY DEFAULT nextval('id_commit'),
	        repository_name TEXT,
	        commit_url TEXT,
	        commit_message TEXT,
	    )
	''')

	cursor.execute('''
	    CREATE TABLE IF NOT EXISTS gh_issues (
	        id INTEGER PRIMARY KEY DEFAULT nextval('id_issues'),
	        repo_name TEXT,
	        issue_url TEXT,
	        issue_title TEXT,
	        issue_body TEXT,
	        pr_from_issue TEXT,
	        user_login TEXT,
	        issue_number INTEGER,
	    )
	''')

	cursor.execute('''
	    CREATE TABLE IF NOT EXISTS gh_discussions (
	        id INTEGER PRIMARY KEY DEFAULT nextval('id_discussion'),
	        repo_name TEXT,
	        discussion_number INTEGER,
	        discussion_title TEXT,
	        discussion_body TEXT,
	        author_login TEXT,
	    )
	''')

	cursor.execute('''
	    CREATE TABLE IF NOT EXISTS gh_comments (
	        id INTEGER PRIMARY KEY DEFAULT nextval('id_comment'),
	        repo_name TEXT,
	        discussion_number INTEGER,
	        author_login TEXT,
	        comment_body TEXT,
	    )
	''')
	conn.commit()
	return conn


if __name__ == "__main__":
	con = get_repo_db()
	mine_repos(con, disable_commits=True)
