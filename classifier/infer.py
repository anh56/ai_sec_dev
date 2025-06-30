import argparse
import json
import os
import random
from enum import StrEnum

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import time

print(torch.cuda.is_available())
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
tqdm.pandas()

ID_GH: list[str] = ["repo_name", "discussion_number"]
ID_HF: list[str] = ["model_id", "num"]


class InputType(StrEnum):
    Github = "gh"
    Huggingface = "hf"
    GhIssue = "gh_issue"
    All = "all"


class PredefinedCol(StrEnum):
    IsSecurityPred = "is_security_prediction"
    ProbSigmoid = "prob_sigmoid"
    ProbSoftmax = "prob_softmax"
    IsSecurityPredSoftmax = "is_security_prediction_softmax"
    IsSecurityPredSigmoid = "is_security_prediction_sigmoid"
    GithubDiscussionTitle = "discussion_title"
    GithubDiscussionBody = "discussion_body"
    GithubDiscussionComment = "comment_body"
    HuggingfaceDiscussionTitle = "title"
    HuggingfaceDiscussionComment = "content"
    GithubIssueTitle = "issue_title"
    GithubIssueBody = "issue_body"
    AllContent = "content"


class OutputFile(StrEnum):
    RawPrediction = "raw_predictions.csv"
    AggregatedPrediction = "aggregated_predictions.csv"


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f"Loaded tokenizer and model from: {model_path}")
    return tokenizer, model


def classify_comment(tokenizer, model, device, text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    classification = torch.argmax(logits, dim=-1).item()

    # probability of "security"
    prob_sigmoid = torch.sigmoid(logits[:, 1] - logits[:, 0]).item()
    prob_softmax = torch.nn.functional.softmax(logits, dim=-1)[:, 1].item()
    return classification, prob_sigmoid, prob_softmax


def run_inference(
    experiment_name: str,
    input_file_name: str,
    input_folder: str,
    output_folder: str,
    model_folder: str,
    input_type: InputType = InputType.Github,
    device_str: str = "0",
):
    """Runs inference on the input file using the loaded model."""
    start_time = time.perf_counter()

    output_path = f"{output_folder}/{experiment_name}"
    os.makedirs(output_path, exist_ok=True)

    input_path = f"{input_folder}/{input_file_name}"
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: input file not found at {input_path}")
        return

    print(f"{input_path=},\n{len(df)=}")

    # if input_type == InputType.Github:
    #     if not all(col in df.columns for col in
    #                [PredefinedCol.GithubDiscussionTitle, PredefinedCol.GithubDiscussionBody,
    #                 PredefinedCol.GithubDiscussionComment]):
    #         raise ValueError(
    #             f"Missing columns {PredefinedCol.GithubDiscussionTitle}, {PredefinedCol.GithubDiscussionBody}, {PredefinedCol.GithubDiscussionComment}")
    # elif input_type == InputType.Huggingface:
    #     if not all(col in df.columns for col in
    #                [PredefinedCol.HuggingfaceDiscussionTitle, PredefinedCol.HuggingfaceDiscussionComment]):
    #         raise ValueError(
    #             f"Missing columns {PredefinedCol.HuggingfaceDiscussionTitle}, {PredefinedCol.HuggingfaceDiscussionComment}")
    # elif input_type == InputType.All:
    #     if PredefinedCol.AllContent not in df.columns.tolist():
    #         raise ValueError(f"Missing column {PredefinedCol.AllContent}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", int(device_str))
    print(f"Using device: {device}")

    try:
        tokenizer, model = load_model(model_folder, device)
    except Exception as e:
        print(f"Error when loading the model from {model_folder}")
        print(e)
        return

    df["full_comment"] = None
    if input_type == InputType.Github:
        df["full_comment"] = df[PredefinedCol.GithubDiscussionTitle].astype(str) + df[
            PredefinedCol.GithubDiscussionBody].astype(str) + df[PredefinedCol.GithubDiscussionComment].astype(str)
    elif input_type == InputType.Huggingface:
        df["full_comment"] = df[PredefinedCol.HuggingfaceDiscussionTitle].astype(str) + df[
            PredefinedCol.HuggingfaceDiscussionComment].astype(str)
    elif input_type == InputType.Huggingface:
        df["full_comment"] = df[PredefinedCol.GithubIssueTitle].astype(str) + df[
            PredefinedCol.GithubIssueBody].astype(str)
    elif input_type == InputType.All:
        df["full_comment"] = df[PredefinedCol.AllContent]
    else:
        raise ValueError("Invalid input type")

    df[
        [
            PredefinedCol.IsSecurityPred,
            PredefinedCol.ProbSigmoid,
            PredefinedCol.ProbSoftmax,
        ]
    ] = df["full_comment"].fillna("").progress_apply(
        lambda x: pd.Series(classify_comment(tokenizer, model, device, x))
    )

    print("Exported raw_predictions to file.")
    df.to_csv(f"{output_path}/{OutputFile.RawPrediction}", index=False)

    discussion_aggregation = df.groupby(
        ID_GH if input_type == InputType.Github else ID_HF
    ).agg(
        mean_softmax=(PredefinedCol.ProbSigmoid, "mean"),
        max_softmax=(PredefinedCol.ProbSigmoid, "max"),
        count_softmax=(PredefinedCol.ProbSigmoid, "count"),
        mean_sigmoid=(PredefinedCol.ProbSoftmax, "mean"),
        max_sigmoid=(PredefinedCol.ProbSoftmax, "max"),
        count_sigmoid=(PredefinedCol.ProbSoftmax, "count"),
    ).reset_index()

    # majority voting from comments -> discussion
    discussion_aggregation[PredefinedCol.IsSecurityPredSoftmax] = (
        discussion_aggregation["mean_softmax"] > 0.5
    ).astype(int)

    discussion_aggregation[PredefinedCol.IsSecurityPredSigmoid] = (
        discussion_aggregation["mean_sigmoid"] > 0.5
    ).astype(int)

    print("Exported aggregated_predictions to file.")
    discussion_aggregation.to_csv(
        f"{output_path}/{OutputFile.AggregatedPrediction}",
        index=False
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference completed in {elapsed_time} seconds")

    security_comments = len(df[df[PredefinedCol.IsSecurityPred] == 1])
    security_discussions_sigmoid = len(
        discussion_aggregation[discussion_aggregation[PredefinedCol.IsSecurityPredSigmoid] == 1])
    security_discussions_softmax = len(
        discussion_aggregation[discussion_aggregation[PredefinedCol.IsSecurityPredSoftmax] == 1])
    result = {
        "time_taken": elapsed_time,
        "security_comments": security_comments,
        "total_comments": len(df),
        "ratio_comments": security_comments / len(df),
        "security_discussion_sigmoid": security_discussions_sigmoid,
        "security_discussions_softmax": security_discussions_softmax,
        "total_discussions": len(discussion_aggregation),
        "ratio_discussions_sigmoid": security_discussions_sigmoid / len(discussion_aggregation),
        "ratio_discussions_softmax": security_discussions_softmax / len(discussion_aggregation),
    }
    print(result)
    output_json_path = f"{output_path}/metrics.json"
    json.dump(result, open(f"{output_json_path}", 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a CSV file using a trained model."
    )
    parser.add_argument(
        "-N",
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-F",
        "--input_file_name",
        type=str,
        required=True,
        help="Name of the input CSV file (e.g., new_comments.csv)",
    )
    parser.add_argument(
        "-I",
        "--input_path",
        type=str,
        required=True,
        help="Path to the folder of the input CSV file.",
    )
    parser.add_argument(
        "-O",
        "--output_path",
        type=str,
        required=True,
        help="Path to the folder storing the output files.",
    )
    parser.add_argument(
        "-M",
        "--model_path",
        type=str,
        required=True,
        help="Path to the folder of the trained model (containing config.json, pytorch_model.bin).",
    )
    parser.add_argument(
        "-T",
        "--type",
        type=str,
        choices=[InputType.Github, InputType.Huggingface, InputType.All],
        required=True,
        help="Type of input data (gh or hf).",
    )
    parser.add_argument(
        "-D",
        "--device",
        type=str,
        required=False,
        default="0",
        help="Type of input data (gh or hf).",
    )

    args = parser.parse_args()

    run_inference(
        experiment_name=args.experiment_name,
        input_file_name=args.input_file_name,
        input_folder=args.input_path,
        output_folder=args.output_path,
        model_folder=args.model_path,
        input_type=args.type,
        device_str=args.device,
    )
