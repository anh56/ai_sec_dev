import argparse
import json
import os
import random
from enum import StrEnum

from anyio.lowlevel import checkpoint
from ray import tune
from ray.tune import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sympy.core.random import shuffle
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW, SGD
import torch
import pandas as pd
import optuna

from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	matthews_corrcoef,
	confusion_matrix,
	classification_report
)
import numpy as np
import time

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

print(torch.cuda.is_available())
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
tqdm.pandas()

ID_GH: list[str] = ["repo_name", "discussion_number"]
ID_HF: list[str] = ["model_id", "num"]

# ray cannot handle enum
InputType_Github = "gh"
InputType_Huggingface = "hf"
InputType_GhIssue = "gh_issue"
InputType_All = "all"

PredefinedCol_IsSecurity = "is_security"
PredefinedCol_IsSecurityPred = "is_security_prediction"
PredefinedCol_ProbSigmoid = "prob_sigmoid"
PredefinedCol_ProbSoftmax = "prob_softmax"
PredefinedCol_IsSecurityPredSoftmax = "is_security_prediction_softmax"
PredefinedCol_IsSecurityPredSigmoid = "is_security_prediction_sigmoid"
PredefinedCol_GithubDiscussionTitle = "discussion_title"
PredefinedCol_GithubDiscussionBody = "discussion_body"
PredefinedCol_GithubDiscussionComment = "comment_body"
PredefinedCol_HuggingfaceDiscussionTitle = "title"
PredefinedCol_HuggingfaceDiscussionComment = "content"
PredefinedCol_GithubType = "gh"
PredefinedCol_HuggingfaceType = "hf"
PredefinedCol_AllContent = "content"
PredefinedCol_AllType = "type"
PredefinedCol_GithubIssueTitle = "issue_title"
PredefinedCol_GithubIssueBody = "issue_body"

OutputFile_RawPrediction = "raw_predictions.csv"
OutputFile_AggregatedPrediction = "aggregated_predictions.csv"
OutputFile_ModelEvaluation = "model_evaluation.txt"
OutputFile_ModelReport = "model_report.txt"
OutputFile_ModelTrain = "model_train.txt"
OutputFile_ModelTest = "model_test.txt"
OutputFile_ModelMetrics = "metrics.csv"


class CommentDataset(Dataset):
	def __init__(
		self,
		dataframe,
		tokenizer,
		max_length,
		input_type: str
	):
		self.dataframe = dataframe
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.input_type = input_type
		self.len = len(dataframe)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		row = self.dataframe.iloc[index]

		label = row[PredefinedCol_IsSecurity]

		comment_body = None
		if self.input_type == InputType_Github:
			comment_body = (
				str(row[PredefinedCol_GithubDiscussionTitle]) + " " +
				str(row[PredefinedCol_GithubDiscussionBody]) + " " +
				str(row[PredefinedCol_GithubDiscussionComment]
					)
			)
		elif self.input_type == InputType_Huggingface:
			comment_body = (
				str(row[PredefinedCol_HuggingfaceDiscussionTitle]) + " " +
				str(row[PredefinedCol_HuggingfaceDiscussionComment])
			)
		elif self.input_type == InputType_GhIssue:
			comment_body = (
				str(row[PredefinedCol_GithubIssueTitle]) + " " +
				str(row[PredefinedCol_GithubIssueBody])
			)

		elif self.input_type == InputType_All:
			comment_body = str(row["content"])
		else:
			raise ValueError("Invalid input type")

		if not comment_body:
			raise ValueError("Empty input")

		encoding = self.tokenizer(
			comment_body,
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
			add_special_tokens=True
		)

		item = {
			"input_ids": encoding["input_ids"].squeeze(0),
			"attention_mask": encoding["attention_mask"].squeeze(0),
			"labels": torch.tensor(label, dtype=torch.long)
		}
		return item


MAPPING = {
	"secbert": ("jackaduma/SecBERT", "jackaduma/SecBERT"),
	"secroberta": ("jackaduma/SecRoBERTa", "jackaduma/SecRoBERTa"),
	"securebert": ("ehsanaghaei/SecureBERT_Plus", "ehsanaghaei/SecureBERT_Plus"),
	"distilbert": ("distilbert-base-uncased", "distilbert-base-uncased"),
	"bert_base": ("google-bert/bert-base-uncased", "google-bert/bert-base-uncased"),
	"bert_large": ("google-bert/bert-large-uncased", "google-bert/bert-large-uncased"),
	"roberta_base": ("FacebookAI/roberta-base", "FacebookAI/roberta-base"),
}


def load_model(model_mapping, device):
	tokenizer_name = MAPPING[model_mapping][0]
	model_name = MAPPING[model_mapping][1]
	print(f"{model_mapping=} ,{tokenizer_name=}, {model_name=}")
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	model.to(device)
	model.eval()

	print(f"{tokenizer=}")
	print(f"{model=}")
	return tokenizer, model


def train_model(model, train_loader, optimizer, scheduler, device):
	model.train()
	total_loss = 0

	for batch in tqdm(train_loader, desc="Training", leave=False):
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)

		optimizer.zero_grad()
		outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs.loss
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	return total_loss / len(train_loader)


def evaluate_model(model, val_loader, device):
	model.eval()
	all_preds = []
	all_labels = []
	total_loss = 0
	loss_fn = torch.nn.CrossEntropyLoss()

	with torch.no_grad():
		for batch in tqdm(val_loader, desc="Evaluating", leave=False):
			batch = {k: v.to(device) for k, v in batch.items()}

			outputs = model(**batch)
			logits = outputs.logits
			preds = torch.argmax(logits, dim=-1).cpu().numpy()
			labels = batch['labels'].cpu().numpy()

			loss = loss_fn(logits, batch['labels'])
			total_loss += loss.item()

			all_preds.extend(preds)
			all_labels.extend(labels)
	avg_val_loss = total_loss / len(val_loader)

	return all_preds, all_labels, avg_val_loss


def write_metric(
	path,
	y_true,
	y_pred,
	start_time,
	fold: int | None = None,
	epoch: int | None = None,
	train_loss: float | None = None,
	eval_loss: float | None = None,
	test_loss: float | None = None,
	return_json: bool = False
):
	metrics = {
		"f1_macro": f1_score(y_true, y_pred, zero_division=0, average="macro"),
		"mcc": matthews_corrcoef(y_true, y_pred),
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1": f1_score(y_true, y_pred, zero_division=0),
		"fold": fold,
		"epoch": epoch,
		"train_loss": train_loss,
		"eval_loss": eval_loss,
		"test_loss": test_loss,
		"elapsed_time": time.perf_counter() - start_time,
	}
	conf_matrix = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred)

	print(f"{metrics=}")

	with open(f"{path}/{OutputFile_ModelReport}", "a") as f:
		f.write(f"Epoch:  {epoch}\n")
		f.write("Classification Report:\n")
		f.write(report + "\n\n")
		f.write("Confusion Matrix:\n")
		np.savetxt(f, conf_matrix, fmt="%d", delimiter="\t")

	metric_path = f"{path}/{OutputFile_ModelMetrics}"
	metrics_df = pd.DataFrame([metrics])
	if os.path.exists(metric_path):
		previous_df = pd.read_csv(metric_path)
		metrics_df = pd.concat([previous_df, metrics_df], ignore_index=True)
	metrics_df.to_csv(metric_path, index=False)
	if return_json:
		return metrics


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


def train_tune(
	config,
	model_mapping,
	train_df,
	input_type,
	skf,
	device,
	epochs,
	output_path,
	start_time,
	patience,
):
	for fold, (train_index, val_index) in enumerate(
		skf.split(train_df, train_df[PredefinedCol_IsSecurity]), 1
	):
		fold_train_df = train_df.iloc[train_index]
		fold_val_df = train_df.iloc[val_index]

		tokenizer, model = load_model(model_mapping, device)

		train_dataset = CommentDataset(fold_train_df, tokenizer, max_length=512, input_type=input_type)
		val_dataset = CommentDataset(fold_val_df, tokenizer, max_length=512, input_type=input_type)

		train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]))

		# Optimizer Selection
		if config["optimizer"] == "adamw":
			optimizer = AdamW(
				model.parameters(),
				lr=config["lr"],
				weight_decay=config["weight_decay"]
			)
		else:  # SGD
			optimizer = SGD(
				model.parameters(),
				lr=config["lr"],
				weight_decay=config["weight_decay"],
				momentum=0.9
			)

		# Scheduler Setup
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode='min',  # Minimize validation loss
			factor=config["reduce_lr_factor"],  # Learning rate reduction factor
			patience=config["reduce_lr_patience"],  # Patience epochs
		)

		epochs_no_improve = 0
		best_mcc = -1

		# early stopping in each fold
		for epoch in range(epochs):
			print(f"Epoch {epoch + 1}")
			train_loss = train_model(model, train_loader, optimizer, scheduler, device)
			print(f"Train loss: {train_loss:.4f}")

			# Evaluation
			preds, labels, eval_loss = evaluate_model(model, val_loader, device)

			# reducelr runs on eval loss
			scheduler.step(eval_loss)

			metrics = write_metric(
				path=output_path,
				y_true=labels,
				y_pred=preds,
				start_time=start_time,
				fold=fold,
				epoch=epoch,
				train_loss=train_loss,
				eval_loss=eval_loss,
				return_json=True
			)

			tune.report(metrics,)
			mcc = metrics["mcc"]
			print(f"Validation MCC: {mcc:.4f}")
			if mcc > best_mcc:
				best_mcc = mcc
				epochs_no_improve = 0
				model.save_pretrained(f"{output_path}/{fold}/best_model")
				tokenizer.save_pretrained(f"{output_path}/{fold}/best_model")
				# tune.report(
				# 	metrics,
				# 	checkpoint=Checkpoint.from_directory(
				# 		f"{output_path}/{fold}/best_model"
				# 	)
				# )
			else:
				epochs_no_improve += 1

			if epochs_no_improve >= patience:
				print("Early stopping")
				break


def run_classifier(
	experiment_name: str,
	input_file_name: str,
	input_folder: str,
	output_folder: str,
	model_name: str,
	input_type: str = InputType_Github,
	epochs: int = 10,
	k_folds: int = 10,
	patience: int = 3,
	device_no: str = "0",
):
	start_time = time.perf_counter()

	# split the name and create a new folder in the output folder
	# input_file = input_file_name.split(".")[0]
	output_path = f"{output_folder}/{experiment_name}"
	os.makedirs(output_path, exist_ok=True)
	input_path = f"{input_folder}/{input_file_name}"

	# model init
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda", int(device_no))

	print(f"Using device: {device}, model name: {model_name}")
	# tokenizer, model = load_model(model_name, device)

	# dataset init
	df = pd.read_csv(input_path)
	print(f"{input_path=},\n{len(df)=}")

	# 80/10/10
	stratify_cols = [PredefinedCol_IsSecurity]
	if input_type == InputType_All:
		stratify_cols = [PredefinedCol_IsSecurity, PredefinedCol_AllType]

	train_df, test_df = train_test_split(
		df,
		test_size=0.1,
		random_state=42,
		# shuffle=True,
		stratify=df[stratify_cols]
	)

	print(
		f"Train size: {len(train_df)}, "
		f"Test size: {len(test_df)}"
	)

	optuna_search = OptunaSearch(
		{
			"lr": optuna.distributions.LogUniformDistribution(1e-5, 1e-3),  # Learning rate
			"weight_decay": optuna.distributions.LogUniformDistribution(1e-3, 1e-1),  # Weight decay
			"batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32, 64]),  # Batch size
			"optimizer": optuna.distributions.CategoricalDistribution(["adamw", "sgd"]),  # Optimizer
			"reduce_lr_factor": optuna.distributions.CategoricalDistribution([0.1, 0.2, 0.5]),
			"reduce_lr_patience": optuna.distributions.CategoricalDistribution([3, 5, 7]),
		},
		metric="mcc",
		mode="max"
	)
	concurrence_search = ConcurrencyLimiter(optuna_search, max_concurrent=2)

	# kfold on train + val
	# fold should be the same for all models so that all models are trained on the same fold data

	skf = StratifiedKFold(
		n_splits=k_folds,
		shuffle=True,
		random_state=42,
	)
	# nested CV
	analysis = tune.run(
		tune.with_parameters(
			# config, model_mapping, train_dataset, val_dataset, device, epochs=10
			train_tune,
			model_mapping=model_name,
			train_df=train_df,
			skf=skf,
			input_type=input_type,
			device=device,
			epochs=epochs,
			output_path=output_path,
			start_time=start_time,
			patience=patience,
		),
		# config=config,
		search_alg=concurrence_search,
		num_samples=20,
		# param_space=search_spaces,
		resources_per_trial={"cpu": 12, "gpu": 1},
		# storage_path=f"./{output_path}/tune_results/fold_{fold}",
		metric="mcc",
		mode="max",
	)

	print(f'Overall Best Trial: {analysis.get_best_config("mcc", "max")} \n')
	analysis.results_df.to_csv(f"{output_path}/tune_results.csv", index=False)
	analysis.best_result_df.to_csv(f"{output_path}/best_tune_results.csv", index=False)
	# print(analysis.best_checkpoint)

	# Get best hyperparameters and metrics from the fold
	best_trial = analysis.get_best_trial("mcc", "max", "last")
	best_config = best_trial.config
	best_mcc = best_trial.last_result["mcc"]
	print(f"Best MCC: {best_mcc}, {best_config}, {best_trial}")


	# # CV on entire set
	# for fold, (train_index, val_index) in enumerate(
	# 	skf.split(train_df, train_df[PredefinedCol_IsSecurity]), 1
	# ):
	# 	# Train a final model with best_config
	final_tokenizer, final_model = load_model(model_name, device)

	best_mcc = -1
	epochs_no_improve = 0

	final_train_dataset = CommentDataset(train_df, final_tokenizer, max_length=512, input_type=input_type)
	final_val_dataset = CommentDataset(test_df, final_tokenizer, max_length=512, input_type=input_type)
	final_train_loader = DataLoader(final_train_dataset, batch_size=int(best_config["batch_size"]), shuffle=True)
	final_val_loader = DataLoader(final_val_dataset, batch_size=int(best_config["batch_size"]))

	if best_config["optimizer"] == "adamw":
		optimizer = AdamW(
			final_model.parameters(),
			lr=best_config["lr"],
			weight_decay=best_config["weight_decay"]
		)
	else:  # SGD
		optimizer = SGD(
			final_model.parameters(),
			lr=best_config["lr"],
			weight_decay=best_config["weight_decay"],
			momentum=0.9
		)
	scheduler = ReduceLROnPlateau(
		optimizer,
		mode='min',  # Minimize validation loss
		factor=best_config["reduce_lr_factor"],  # Learning rate reduction factor
		patience=best_config["reduce_lr_patience"],  # Patience epochs
	)

	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}")
		train_loss = train_model(final_model, final_train_loader, optimizer, scheduler, device)
		print(f"Train loss: {train_loss:.4f}")

		# Evaluation
		preds, labels, eval_loss = evaluate_model(final_model, final_val_loader, device)

		# reducelr runs on eval loss
		scheduler.step(eval_loss)

		write_metric(
			path=output_path,
			y_true=labels,
			y_pred=preds,
			start_time=start_time,
			epoch=epoch,
			train_loss=train_loss,
			eval_loss=eval_loss,
		)

		mcc = matthews_corrcoef(labels, preds)
		print(f"Validation MCC: {mcc:.4f}")
		if mcc > best_mcc:
			best_mcc = mcc
			epochs_no_improve = 0
			final_model.save_pretrained(f"{output_path}/best_model")
			final_tokenizer.save_pretrained(f"{output_path}/best_model")
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= patience:
			print("Early stopping")
			break

	print("Finished Training!")
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	print(f"Training completed in {elapsed_time} seconds")

	print("\nTesting Phase")

	# Use the best model for evaluating the test dataset
	model = AutoModelForSequenceClassification.from_pretrained(f"{output_path}/best_model").to(device)
	tokenizer = AutoTokenizer.from_pretrained(f"{output_path}/best_model")

	test_dataset = CommentDataset(test_df, tokenizer, max_length=512, input_type=input_type)
	test_loader = DataLoader(test_dataset, batch_size=int(best_config["batch_size"]))
	preds, labels, test_loss = evaluate_model(model, test_loader, device)
	write_metric(
		path=output_path,
		y_true=labels,
		y_pred=preds,
		start_time=start_time,
		test_loss=test_loss
	)

	# also export the test df
	test_df[PredefinedCol_IsSecurityPred] = preds
	test_df.to_csv(f"{output_path}/test_df.csv", index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Run the classifier on the input CSV and output predictions."
	)
	parser.add_argument(
		"-N",
		"--experiment_name",
		type=str,
		required=True,
		help="Name of the folder to store results",
	)
	parser.add_argument(
		"-F",
		"--input_file_name",
		type=str,
		required=True,
		help="Name of the input CSV file (e.g., merged_gh_discussions.csv)",
	)
	parser.add_argument(
		"-I",
		"--input_path",
		type=str,
		required=True,
		help="Path to the folder of the CSV file (e.g., merged_after_manual/merged_gh_discussions.csv)",
	)
	parser.add_argument(
		"-O",
		"--output_path",
		type=str,
		required=True,
		help="Path to the folder storing the output file (e.g., prediction/)",
	)

	parser.add_argument(
		"-T",
		"--type",
		type=str,
		choices=[InputType_Github, InputType_Huggingface, InputType_GhIssue, InputType_All],
		required=True,
		help="Path to store the output CSV file (e.g., prediction/merged_gh_discussions.csv)",
	)

	parser.add_argument(
		"-E",
		"--epochs",
		type=int,
		required=False,
		default=10,
		help="Number of epochs. Default: 10",
	)

	parser.add_argument(
		"-KF",
		"--k_folds",
		type=int,
		required=False,
		default=10,
		help="Number of KFolds. Default: 10",
	)
	parser.add_argument(
		"-P",
		"--patience",
		type=int,
		required=False,
		default=3,
		help="Number of patience epoch. Default: 3",
	)
	parser.add_argument(
		"-M",
		"--model",
		type=str,
		required=False,
		help="Number of KFolds. Default: 10",
	)
	parser.add_argument(
		"-D",
		"--device",
		type=str,
		required=False,
		default="0",
		help="GPU to use",
	)
	args = parser.parse_args()

	print(f"{args=}")

	run_classifier(
		experiment_name=args.experiment_name,
		input_file_name=args.input_file_name,
		input_folder=args.input_path,
		output_folder=args.output_path,
		input_type=args.type,
		epochs=args.epochs,
		k_folds=args.k_folds,
		patience=args.patience,
		model_name=args.model,
		device_no=args.device,
	)
