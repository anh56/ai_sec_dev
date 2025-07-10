import json
import pickle
import re
import csv
import sys
import duckdb

csv.field_size_limit(sys.maxsize)

from huggingface_hub import HfApi, list_models, ModelCard, errors
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

api = HfApi(token="")


def create_and_get_db_conn():
    con = duckdb.connect("./result/models.db")
    con.sql(
        """
        CREATE TABLE IF NOT EXISTS hf_discussions(
        model_id VARCHAR,
        num INTEGER,
        title VARCHAR,
        git_ref VARCHAR, 
        url VARCHAR,
        keywords JSON,
        );
        """
    )

    con.sql(
        """
        CREATE TABLE IF NOT EXISTS hf_discussion_events(
        model_id VARCHAR,
        num INTEGER,
        event_id VARCHAR,
        event_type VARCHAR,
        content VARCHAR,
        keywords JSON,
        );
        """
    )

def get_distinct_models(db_conn):
    distinct_models = db_conn.execute("SELECT distinct(model_id) FROM models").fetchall()
    distinct_models = [model[0] for model in distinct_models]
    print(f"got {len(distinct_models)} models")
    return distinct_models


def mine_repo_discussions_comments(model, db_conn, corasick_auto):
    print(f"mining {model}")

    try:
        discussions = api.get_repo_discussions(
            repo_id=model,
        )
        for discussion in discussions:
            keywords = set()
            for (
                end_index, (insert_order, original_value)
            ) in corasick_auto.iter(discussion.title.lower()):
                keywords.add(original_value)

            keywords = list(keywords) if keywords else None

            db_conn.execute(
                """
                INSERT INTO hf_discussions VALUES (
                    ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    model,
                    discussion.num,
                    discussion.title,
                    discussion.git_reference,
                    discussion.url,
                    keywords
                )
            )



            discussion_details = api.get_discussion_details(
                repo_id=model,
                discussion_num=discussion.num
            )

            events = discussion_details.events
            for event in events:
                keywords = set()
                content = None
                if event.type == "comment":
                    content = event.content
                if event.type == "commit":
                    content = event.summary

                if content:
                    for (
                        end_index, (insert_order, original_value)
                    ) in corasick_auto.iter(content.lower()):
                        keywords.add(original_value)

                keywords = list(keywords) if keywords else None

                db_conn.execute(
                    """
                    INSERT INTO hf_discussion_events VALUES (
                        ?, ?, ?, ?, ?, ?
                    );
                    """,
                    (
                        model,
                        discussion.num,
                        event.id,
                        event.type,
                        content,
                        keywords
                    )
                )

    except Exception as e:
        print(e)
        print(f"Failed when extracting discussions from {model}.")


if __name__ == "__main__":

    db_conn = create_and_get_db_conn()
    models = get_distinct_models(db_conn)

    corasick_auto = None
    with open('corasick.pkl', 'rb') as corasick_file:
        corasick_auto = pickle.load(corasick_file)

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [
            executor.submit(
                mine_repo_discussions_comments,
                model,
                db_conn,
                corasick_auto,
            ) for model in models
        ]
        for future in as_completed(futures):
            future.result()

    print(f"mined {len(models)} models")