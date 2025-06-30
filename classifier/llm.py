import argparse
import json
import os
import time
from enum import StrEnum
from tqdm import tqdm

from huggingface_hub import InferenceClient
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from concurrent.futures import ThreadPoolExecutor

from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from transformers import pipeline
import pandas as pd

TEMPLATE_GH = """
This is a discussion on GitHub. Classify the discussion as related to security or not.
Response with 0 for non-security and 1 for security, and no other classification values are allowed.
Provide a brief explanation for the classification.

Discussion Title: {title}
Discussion Body: {body}
Comment Body: {comment}
"""

TEMPLATE_HF = """
This is a discussion on HuggingFace. Classify the discussion as related to security or not.
Response with 0 for non-security and 1 for security, and no other classification values are allowed.
Provide a brief explanation for the classification.

Title: {title}
Content: {content}
"""

TEMPLATE_ISSUE = """
This is an issue on GitHub. Classify the issue as related to security or not.
Response with 0 for non-security and 1 for security, and no other classification values are allowed.
Provide a brief explanation for the classification.

Issue Title: {title}
Issue Body: {body}
"""

TEMPLATE_ALL = """
The following content is taken from discussion/issue raised on Github/HuggingFace.
Classify it as related to security or not.
Response with 0 for non-security and 1 for security, and no other classification values are allowed.
Provide a brief explanation for the classification.

Content: {content}
"""

TEMPLATES = {
    "gh": (TEMPLATE_GH, ["title", "body", "comment"]),
    "hf": (TEMPLATE_HF, ["title", "content"]),
    "issues": (TEMPLATE_ISSUE, ["title", "body"]),
    "all": (TEMPLATE_ALL, ["content"]),
}

MODEL_MAPPING = {
    "llama31_8b": "llama3.1:8b",
    "llama33_70b_awq": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    "llama33_70b": "llama3.3:70b",
    "deepseekr1_8b": "deepseek-r1:8b",
    "deepseekr1_70b": "deepseek-r1:70b",
    "deepseekv3": "deepseek-v3:latest",
    "phi4": "phi4:14b",
    "mistral_small": "mistral-small:24b",
}


class ClassificationResponse(BaseModel):
    """
    Structured output for the security classification task.
    Return 1 for security and 0 for non-security.
    Also provide an explanation for the classification.
    """

    classification: int = Field(
        ...,
        description=(
            """
            The classification of the text as security or non-security.
            1 for security and 0 for non-security.
            """
        ),
        # only 0 and 1
        ge=0,
        le=1,
    )
    explanation: str = Field(
        ...,
        description="The explanation of the classification, why the text is security related or not"
    )


def init_llm_runnable(model_name: str) -> Runnable:
    # use ollama
    llm = ChatOllama(
        model=MODEL_MAPPING[model_name],
        temperature=0.0,
        base_url=os.getenv("LLM_BASE_URL"),
    )
    # use vllm
    # llm = ChatOpenAI(
    #     model=MODEL_MAPPING[model_name],
    #     temperature=0.0,
    #     base_url=os.getenv("LLM_BASE_URL"),
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     max_tokens=1000,
    # )
    print(os.getenv("LLM_BASE_URL"))
    print(llm)
    return llm.with_structured_output(ClassificationResponse, method="json_schema")


def run_llm(
    llm: Runnable,
    prompt: str
) -> (ClassificationResponse, str):
    start_time = time.perf_counter()
    response = llm.invoke(prompt)
    end_time = time.perf_counter()
    return response, f"{end_time - start_time:.2f}"


def run_nli_inference(
    pipe,
    row_content: str
):
    # cols = []
    # if type == "gh":
    #     cols = ["discussion_title", "discussion_body", "comment_body"]
    # elif type == "hf":
    #     cols = ["title", "content"]
    # else:
    #     raise ValueError("Invalid type")
    # pipe = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification", device=-1)
    # for index, row in df.iterrows():
    #     row_content = ""
    #     for col in cols:
    #         row_content += str(row[col]) + " "
    #         prediction = run_inference(pipe, row_content)
    #
    #     df.at[index, "is_security_prediction"] = prediction["labels"][0]
    #     df.at[index, "prediction_score"] = prediction["scores"][0]
    #     print(index, prediction["labels"][0], prediction["scores"][0])
    # total = len(df)
    # security = len(df[df["is_security_prediction"] == "security"])
    # non_security = len(df[df["is_security_prediction"] == "non-security"])
    # print(f"Total: {total}, Security: {security}, Non-Security: {non_security}")
    # print(f"Security Percentage: {security / total * 100}%, Non-Security Percentage: {non_security / total * 100}%")
    # security = len(df[df["is_security_prediction"] == "security"])
    # non_security = len(df[df["is_security_prediction"] == "non-security"])
    # print(f"Total: {total}, Security: {security}, Non-Security: {non_security}")
    # print(f"Security Percentage: {security / total * 100}%, Non-Security Percentage: {non_security / total * 100}%")
    # df["is_security_prediction"] = df["is_security_prediction"].map(
    #     {"security": 1, "non-security": 0}
    # ).astype("category")

    prediction = pipe(
        row_content,
        candidate_labels=["security", "non-security"]
    )
    return prediction["labels"][0], prediction["scores"][0]


def process_df(df: DataFrame, type: str, llm: Runnable):
    df["is_security_prediction"] = None
    df["explanation"] = None
    df["request_time"] = None

    title = ""
    body = ""
    comment = ""

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # if index != 2755:
        #     print(index)
        #     continue
        prompt = None
        prompt_template = PromptTemplate(
            template=TEMPLATES[type][0],
            input_variables=TEMPLATES[type][1]
        )
        if type == "gh":
            prompt = prompt_template.format(
                title=str(row["discussion_title"]),
                body=str(row["discussion_body"]),
                comment=str(row["comment_body"])
            )
        elif type == "hf":
            prompt = prompt_template.format(
                title=str(row["title"]),
                content=str(row["content"]),
            )
        elif type == "issues":
            prompt = prompt_template.format(
                title=str(row["issue_title"]),
                body=str(row["issue_body"]),
            )
        elif type == "all":
            prompt = prompt_template.format(
                content=str(row["content"]),
            )
        else:
            raise ValueError("Invalid type")

        assert prompt is not None
        classification, time_taken = run_llm(llm, prompt)

        df.at[index, "is_security_prediction"] = int(classification.classification)
        df.at[index, "explanation"] = str(classification.explanation)
        df.at[index, "request_time"] = time_taken

        tqdm.write(f"{index} {classification.classification} {classification.explanation} {time_taken}")

    # for multi processing
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = []
    #     for index, row in df.iterrows():
    #         if type == "gh":
    #             title = row["discussion_title"]
    #             body = row["discussion_body"]
    #             comment = row["comment_body"]
    #         elif type == "hf":
    #             title = row["title"]
    #             body = row["content"]
    #             comment = ""
    #         else:
    #             raise ValueError("Invalid type")
    #
    #         futures.append(executor.submit(run_llm, llm, title, body, comment))
    #
    #     for index, future in enumerate(futures):
    #         classification, time_taken = future.result()
    #
    #         df.at[index, "is_security_prediction"] = int(classification.classification)
    #         df.at[index, "explanation"] = str(classification.explanation)
    #         df.at[index, "request_time"] = time_taken
    #
    #         print(index, classification.classification, classification.explanation, time_taken)

    return df


def calculate_metrics(df, start):
    y_true = df["is_security"].tolist()
    y_pred = df["is_security_prediction"].tolist()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, zero_division=0, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "time_taken": time.perf_counter() - start,
    }

    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


# df = pd.read_csv("./merged_after_manual/merged_hf_discussions.csv")
# df = process_df(df, "hf")
# df.to_csv("./inference/merged_hf_discussions_llama33_70b.csv", index=False)
# calculate_metrics(df)

def run_inference(
    experiment_name: str,
    input_file_name: str,
    input_folder: str,
    output_folder: str,
    type: str,
    model: str,
):
    start = time.perf_counter()
    output_path = f"{output_folder}/{experiment_name}"
    os.makedirs(output_path, exist_ok=True)

    input_path = f"{input_folder}/{input_file_name}"

    llm = init_llm_runnable(model)
    df = pd.read_csv(input_path)

    df = process_df(df, type, llm)
    df.to_csv(f"{output_path}/{experiment_name}.csv", index=False)
    print(f"Saved the output to {output_path}/{experiment_name}.csv")
    metrics = calculate_metrics(df, start)
    output_json_path = f"{output_path}/metrics.json"
    json.dump(metrics, open(f"{output_json_path}", 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-F", "--input_file", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "-I", "--input_folder", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "-O", "--output_folder", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "-N", "--name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "-T", "--type", type=str, required=True, choices=["gh", "hf", "issues", "all"], help="Type of the dataset"
    )
    parser.add_argument(
        "-M", "--model", type=str, required=True, help="model to run inference",
        choices=MODEL_MAPPING.keys(),
    )

    args = parser.parse_args()
    run_inference(
        experiment_name=args.name,
        input_file_name=args.input_file,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        type=args.type,
        model=args.model,
    )

# df = pd.read_csv("./inference/merged_gh_discussions_with_predictions.csv")


# from langchain_core.prompts import PromptTemplate
# from pydantic import BaseModel, Field
# from langchain.chat_models import init_chat_model
# import pandas as pd
#
# llm = init_chat_model(
#     "gemini-2.0-flash-001",
#     model_provider="google_vertexai"
# )
#
#
# class SecurityOutput(BaseModel):
#     """"
#     Structured output for the security classification task.
#     """
#     input_text: str
#     is_security: bool
#
#
# structured_llm = llm.with_structured_output(SecurityOutput)
#
# structured_llm.invoke("Tell me a joke about cats")
#
# llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
#
# # Load the dataset from a file
# file_path = "path/to/your/dataset.csv"
# df = pd.read_csv(file_path)
#
# # Define the zero-shot prompt
# prompt_template = PromptTemplate(
#     template="Classify the following text as related to security or not:\n\nText: {text}\n\nClassification:",
#     input_variables=["text"]
# )
#
#
# # Function to classify text using the LLM
# def classify_text(text):
#     prompt = prompt_template.format(text=text)
#     response = llm(prompt)
#     return response.strip()
#
#
# # Apply the classification to the dataset
# df["SecurityClassification"] = df["TextColumn"].apply(classify_text)
#
# # Save the classified dataset to a new file
# output_file_path = "path/to/your/output_dataset.csv"
# df.to_csv(output_file_path, index=False)
#
# print("Classification completed and saved to", output_file_path)


# client = InferenceClient(api_key="")
#
# messages = [
# 	{ "role": "user", "content": "Tell me a story" }
# ]
#
# stream = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
# 	messages=messages,
# 	temperature=0.5,
# 	max_tokens=8704,
# 	top_p=0.7,
# 	stream=True
# )
#
# for chunk in stream:
#     print(chunk.choices[0].delta.content)
