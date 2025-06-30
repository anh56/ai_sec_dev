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


def get_model_list(
    update_model_list: bool = False,
    write_to: str = "./result/model_list.csv",
    db_conn=None
):
    if update_model_list:
        models = api.list_models(
            sort='likes',
            direction=-1,
            full=True,
            cardData=True
        )

        # keep the list of models in a separate file
        with open(write_to, "a") as f:
            writer = csv.writer(f)
            for model in models:
                writer.writerow(
                    [
                        model.id
                    ]
                )

    # read back
    model_list = []
    with open(write_to, "r") as f:
        model_list = list(f.read().splitlines())

    # or read from db
    if db_conn:
        model_list = db_conn.execute("SELECT model_id FROM model_list WHERE done = False").fetchall()
        model_list = [model[0] for model in model_list]
    print(f"got {len(model_list)} models")

    return model_list


def mine_models(model_list: list[str], write_to: str = "./result/models.csv", corasick=None, db_conn=None):
    count = 0

    for model in model_list:
        count += 1
        try:
            mine_model(model, write_to, corasick, db_conn)
            print(f"mined {model}, {count} models")
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            continue


def mine_model(model: str, write_to: str = "./result/models.csv", corasick=None, db_conn=None):
    print(f"mining {model}")
    # if db_conn:
    #     done = db_conn.execute("SELECT done FROM model_list WHERE model_id = ?", (model,)).fetchone()
    #     if done[0]:
    #         print(f"skipping {model} as it is already done")
    #         return

    # security info needs to be in a separate call
    model_security_info = None
    try:
        model_security_info = api.model_info(
            model,
            securityStatus=True
        )
    except (errors.RepositoryNotFoundError, errors.HfHubHTTPError) as e:
        print(f"Repository {model} not found, exc {e}")
        if db_conn:
            db_conn.execute("UPDATE model_list SET done = ? WHERE model_id = ?", (True, model))
        return

    if not (model_security_info.downloads > 0 or model_security_info.likes > 0):
        print(f"skipping {model} with no downloads or likes")
        if db_conn:
            db_conn.execute("UPDATE model_list SET done = ? WHERE model_id = ?", (True, model))
        return

    # card info and other info
    model_info_expand = api.model_info(
        model,
        expand=[
            "baseModels",
            "childrenModelCount",
            "downloadsAllTime",
            "trendingScore",
            "cardData",
            "tags"
        ]
    )

    card_data_dict = model_info_expand.card_data.to_dict() if model_info_expand.card_data else None
    if card_data_dict:
        card_data_dict = {
            k: v for k, v in card_data_dict.items()
            if not (any(x in k for x in ["extra_gated", "widget"]))  # remove this to save space
        }

    # base model can comes from tags or model cards
    base_model_from_card = None
    if card_data_dict:
        base_model_from_card = card_data_dict.get("base_model")

    # scanner results
    scanner_result = dict(model_security_info.security_repo_status)
    scan_done = scanner_result.get("scansDone")
    # convert to None for easier processing later
    files_with_issues = scanner_result["filesWithIssues"] if scanner_result.get("filesWithIssues") else None

    # model chains
    chains = dict(model_info_expand.childrenModelCount)
    adapter = chains.get("adapter", 0)
    merge = chains.get("merge", 0)
    quantized = chains.get("quantized", 0)
    finetune = chains.get("finetune", 0)

    # readme files
    github_links = None
    kw_in_hf_readme = None

    try:
        readme_path = api.hf_hub_download(
            repo_id=model,
            filename="README.md",
            # cache_dir=f"./huggingface_models_readme/{model}",
            # local_dir=f"./huggingface_models_readme/{model}",
            local_files_only=False
        )
        with open(readme_path, "r") as file:
            content = file.read()

        if content:
            # Find all links in the content that contain the word "github" or are GitHub links
            github_links = re.findall(r'https?:\/\/?github\.com\/[\w-]+\/[\w-]+', content)
            # dedup
            github_links = list(set(github_links))

            # also check for security keywords in the readme
            if corasick:
                kw_in_hf_readme = set()
                for end_index, (insert_order, original_value) in corasick.iter(content.lower()):
                    kw_in_hf_readme.add(original_value)
                kw_in_hf_readme = list(kw_in_hf_readme) if kw_in_hf_readme else None

    except Exception as e:
        print(e)
        print(f"{model} has no readme file")

    with open(write_to, "a") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                model,
                model_security_info.downloads,
                model_info_expand.downloads_all_time,
                model_security_info.likes,
                model_info_expand.trending_score,
                model_security_info.pipeline_tag,
                model_security_info.tags,
                card_data_dict,
                base_model_from_card,
                scan_done,
                files_with_issues,
                adapter,
                merge,
                quantized,
                finetune,
                github_links,
                kw_in_hf_readme
            ]
        )

    if db_conn:
        try:
            db_conn.execute(
                """
                INSERT INTO models VALUES (
                $model_id,$downloads,$downloads_all_time,$likes,$trending_score,
                $pipeline_tags,$tags,$card_data,$base_model_from_card_data,$scan_done,
                $files_with_issues,$adapter_count,$merge_count,$quantized_count,$finetune_count,
                $github_links,$kw_in_hf_readme
                )
                """,
                {
                    "model_id": model,
                    "downloads": model_security_info.downloads,
                    "downloads_all_time": model_info_expand.downloads_all_time,
                    "likes": model_security_info.likes,
                    "trending_score": model_info_expand.trending_score,
                    "pipeline_tags": model_security_info.pipeline_tag,
                    "tags": model_security_info.tags,
                    "card_data": json.dumps(card_data_dict),
                    "base_model_from_card_data": base_model_from_card,
                    "scan_done": scan_done,
                    "files_with_issues": files_with_issues,
                    "adapter_count": adapter,
                    "merge_count": merge,
                    "quantized_count": quantized,
                    "finetune_count": finetune,
                    "github_links": github_links,
                    "kw_in_hf_readme": kw_in_hf_readme,
                }
            )

            db_conn.execute(
                "UPDATE model_list SET done = ? WHERE model_id = ?", (True, model)
            )
            # db_conn.execute("COMMIT")
        except Exception as e:
            print(e)
            print(f"failed to insert {model} into the database")


# csv headers
# model_id, downloads, downloads_all_time, likes, trending_score, pipeline_tags,
# tags, card_data, base_model_from_card_data, scan_done, files_with_issues,
# adapter_count, merge_count, quantized_count, finetune_count, github_links, kw_in_hf_readme

def create_and_get_db_conn():
    con = duckdb.connect("./result/models.db")
    con.sql("CREATE TABLE IF NOT EXISTS model_list (model_id VARCHAR, done BOOLEAN);")
    con.sql("""
        CREATE TABLE IF NOT EXISTS models (
        model_id VARCHAR,
        downloads INTEGER,
        downloads_all_time INTEGER,
        likes INTEGER,
        trending_score FLOAT,
        pipeline_tags VARCHAR,
        tags VARCHAR,
        card_data VARCHAR,
        base_model_from_card_data VARCHAR,
        scan_done BOOLEAN,
        files_with_issues VARCHAR,
        adapter_count INTEGER,
        merge_count INTEGER,
        quantized_count INTEGER,
        finetune_count INTEGER,
        github_links VARCHAR,
        kw_in_hf_readme VARCHAR
        );
    """)

    return con


if __name__ == "__main__":

    db_conn = create_and_get_db_conn()
    model_list = get_model_list(False, db_conn=db_conn)

    corasick_auto = None
    with open('corasick.pkl', 'rb') as corasick_file:
        corasick_auto = pickle.load(corasick_file)

    # mine_models(model_list, corasick=corasick_auto, db_conn=db_conn)
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [
            executor.submit(
                mine_model,
                model,
                "./result/models.csv",
                corasick_auto,
                db_conn
            ) for model in model_list
        ]
        for future in as_completed(futures):
            future.result()

    print(f"mined {len(model_list)} models")
