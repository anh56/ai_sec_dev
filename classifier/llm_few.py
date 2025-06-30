import argparse
import json
import os
import time
from enum import StrEnum
from tqdm import tqdm

from huggingface_hub import InferenceClient
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
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

few_shot_example = [
    {
        "content": "CVE-2007-4559 Patch\n# Patching CVE-2007-4559\nHi, we are security researchers from the Advanced Research Center at [Trellix](https://www.trellix.com). We have began a campaign to patch a widespread bug named CVE-2007-4559. CVE-2007-4559 is a 15 year old bug in the Python tarfile package. By using extract() or extractall() on a tarfile object without sanitizing input, a maliciously crafted .tar file could perform a directory path traversal attack. We found at least one unsantized extractall() in your codebase and are providing a patch for you via pull request. The patch essentially checks to see if all tarfile members will be extracted safely and throws an exception otherwise. We encourage you to use this patch or your own solution to secure against CVE-2007-4559. Further technical information about the vulnerability can be found in this [blog](https://www.trellix.com/en-us/about/newsroom/stories/research/tarfile-exploiting-the-world.html).\nIf you have further questions you may contact us through this projects lead researcher [Kasimir Schulz](mailto:kasimir.schulz@trellix.com).",
        "output": """{"classification": 1, "explanation": "The text explicitly mentions a CVE (CVE-2007-4559), a security vulnerability, and discusses patching it, which is a clear security context."}"""
    },
    {
        "content": "Critical Vulnerability CVE-2024-34359 \n Is there a fix release planned to avoid the CVE for llama versions < v0.2.72 (to support applications in prod built off of ggml arch for instance which might need an older llamacpp python version/older CUDA version",
        "output": """{"classification": 1, "explanation": "The content directly references a Critical Vulnerability and a CVE (CVE-2024-34359), making it security-related."}"""
    },
    {
        "content": "security warning on model.onnx \n Protect AI has raised a security warning about the file model.onnx containing an 'architectural backdoor' susceptible to attack. \n Are you aware of this, and what is your take on it? \n Thanks! \n See: https://protectai.com/insights/models/minishlab/potion-base-8M/dcbec7aa2d52fc76754ac6291803feedd8c619ce/files?blob-id=975e384201f4a58a73772242f4797ab37464bdff&utm_source=huggingface",
        "output": """{"classification": 1, "explanation": "The text discusses a security warning, an architectural backdoor, and a potential attack,of which are core security concepts."}"""
    },
    {
        "content": "Development for Local Machine and for production Server Requirments \n Hello GitHub Community,\nI am in the process of setting up environments for a new project and am seeking advice on best practices and recommendations for managing development and production environments.\nQuestion:\nWhat are the key differences between local machine setups for development and server setups for production environments that I should be aware of? I'm particularly interested in:\nHardware Requirements: What should I consider in terms of CPU, memory, and storage?\nSoftware Dependencies: How should I manage version differences between development and production?\nSecurity Practices: What security measures are critical for production but might be overlooked during development?\nPerformance Optimization: Any tips for ensuring that the production environment is optimized for performance?\nDeployment Practices: Recommendations on deployment strategies (e.g., continuous integration, blue-green deployments)?\nAdditional Context:\n[Include any additional details or specific constraints, such as budget limitations, specific industry regulations, etc.]\nAny insights or resources you could share would be greatly appreciated. Thank you in advance for your help and advice!",
        "output": """{"classification": 0, "explanation": "This content is about general software development practices, environments, and performance, with no mention of specific vulnerabilities or threats."}"""
    },
    {
        "content": "Why is Flax 7x slower than Haiku? \n Here is a minimal, complete, and verifiable example: mcve.zip. You can execute the scripts by \n pipenv install \n pipenv run python3 mcve_haiku.py \n pipenv run python3 mcve_flax.py \n Here is the background: I was trying to convert my research code from Haiku to Flax, but the performance got significantly worse for some reason. Basically mcve_flax.py and mcve_haiku.py perform the same training procedure to a linear neural network (at least that's what I intended), but the former takes 07:23 on my 2017 MacBook Pro, whereas the latter merely takes 01:05. \n Am I doing anything wrong, or is this a known issue?",
        "output": """{"classification": 0, "explanation": "The discussion is focused on performance comparison between two machine learning frameworks (Flax and Haiku) and is not related to security."}"""
    },
    {
        "content": "AX's GitHub repository is moving to the jax-ml GitHub org in September 2024. \n Summary: \n  \n The canonical JAX repository is relocating to github.com/jax-ml/jax. \n URLs and forks of github.com/google/jax will redirect to the new location. \n This will improve our tooling options for continuous integration and testing. \n This does not reflect any change in how JAX is developed (aside from the URL). \n  \n Coming soon, the JAX core repository will relocate from github.com/google/jax to github.com/jax-ml/jax. GitHub takes care to help avoid any interruption due to the move – see the GitHub docs on transferring repositories. In particular, forks and HTTPS/SSH requests will automatically redirect to the new location, indefinitely. \n The JAX core team has operated the jax-ml GitHub organization for several years. It houses a handful of repositories related to the JAX core—such as ml_dtypes and jax-triton—plus a few more experimental projects. Meanwhile, The JAX core repository has been located within the broader Google-wide GitHub org since its initial open-source launch in 2018, before we had jax-ml. \n Our jax-ml GitHub org offers us useful features, especially ones that we can use to improve the GitHub actions that run on every pull request. For instance, org-wide self-hosted runners will give us better test result UI, test coverage across more platforms (including more TPUs and GPUs), extra security features, and better availability (e.g. runners spread geographically). \n We are moving the repository to take advantage of these and any other nice things going forward. We see a bonus opportunity to have the JAX core listed with its closely related extensions in one place. This move does not affect anything else about how JAX is developed. Expect the only change for now to be the URL and improved CI.",
        "output": """{"classification": 0, "explanation": "This is an announcement about a repository migration and infrastructure changes for a project, not a security issue."}"""
    },
]


system_message = """
You are an expert security analyst.
The following content is taken from a discussion or issue raised on Github or HuggingFace.
Your task is to classify the content as related to security or not.
You must respond with a JSON object containing two fields: 'classification' and 'explanation'.
- For 'classification', use 1 for security-related content and 0 for non-security content.
- For 'explanation', provide a brief justification for your classification.
"""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Content: {content}"),
        ("ai", "{output}"),
    ]
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=few_shot_example,
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        few_shot_prompt,
        ("human", "Classify the following content:\nContent: {content}"),
    ]
)

TEMPLATES = {
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





def process_df(df: DataFrame, type: str, llm_chain: Runnable):
    df["is_security_prediction"] = None
    df["explanation"] = None
    df["request_time"] = None

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        input_data = {}
        if type == "all":
            input_data["content"] = str(row["content"])
        else:
            raise ValueError("Invalid type")

        start_time = time.perf_counter()
        # print(llm_chain)
        response = llm_chain.invoke(input_data)
        end_time = time.perf_counter()
        time_taken = f"{end_time - start_time:.2f}"

        df.at[index, "is_security_prediction"] = int(response.classification)
        df.at[index, "explanation"] = str(response.explanation)
        df.at[index, "request_time"] = time_taken

        tqdm.write(f"{index} {response.classification} {response.explanation} {time_taken}")

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
    llm_chain = chat_prompt | llm

    df = pd.read_csv(input_path)

    df = process_df(df, type, llm_chain)
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
