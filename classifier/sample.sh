# remember to get export the content of the corresponding discussion before running
# train
python classifier_tune.py -N "train_distilbert_" -F "all.csv" -I "manual" -O "prediction_tuned" -T "all" -M "distilbert" -E 10 -KF 10

# test and infer use the same script
# test distilbert
python infer.py -N "test_distilbert_best" -F "test.csv" -I "test" -O "inference" -M "prediction_tuned/all_9_train_distilbert/best_model/" -T "all" -D 1

# infer best ditilbert
python infer.py -N "gh_distilbert_best" -F "merged_gh_discussions.csv" -I "merged" -O "inference" -M "prediction_tuned/manual_gh_distilbert/best_model/" -T "gh"
python infer.py -N "hf_distilbert_best" -F "merged_hf_discussions.csv" -I "merged" -O "inference" -M "prediction_tuned/manual_hf_distilbert/best_model/" -T "hf"


# test llm zeroshot
python llm.py -N "test_llama31_8b" -F "test.csv" -I "test" -O "llm" -T "all" -M "llama31_8b"

# test llm fewshot
python llm_few.py -N "test_fewshot_llama31_8b" -F "test.csv" -I "test" -O "llm" -T "all" -M "llama31_8b"