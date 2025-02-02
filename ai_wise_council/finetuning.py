import os
import random
from itertools import product
from huggingface_hub import upload_file
from huggingface_hub import login

import torch

from ai_wise_council import CACHE_DIR
from ai_wise_council.train import run_training

login(token=os.getenv("HF_TOKEN"))

PARAM_GRID = {
    "learning_rate": [1e-5, 5e-5],
    "per_device_train_batch_size": [1, 4],
    "gradient_accumulation_steps": [4, 8],
    "num_train_epochs": [2, 4],
    "warmup_ratio": [0.3, 0.03],
}


def run_finetuning(n_max_combinations: int):
    param_combinations = list(
        product(
            PARAM_GRID["learning_rate"],
            PARAM_GRID["per_device_train_batch_size"],
            PARAM_GRID["gradient_accumulation_steps"],
            PARAM_GRID["num_train_epochs"],
            PARAM_GRID["warmup_ratio"],
        )
    )

    if len(param_combinations) > n_max_combinations:
        param_combinations = random.sample(param_combinations, n_max_combinations)

    print(f"Running {len(param_combinations)} combinations")

    # run combinations

    results = {}
    best_eval_loss = float("inf")

    for combination in param_combinations:
        training_args = {
            "learning_rate": combination[0],
            "per_device_train_batch_size": combination[1],
            "gradient_accumulation_steps": combination[2],
            "num_train_epochs": combination[3],
            "warmup_ratio": combination[4],
        }
        torch.cuda.memory_summary(device=None, abbreviated=False)
        trained_model = run_training(training_args)
        eval_result = trained_model.evaluate()

        results[
            f"lr={combination[0]}, per_device_train_batch_size={combination[1]}, gradient_accumulation_steps={combination[2]}, num_train_epochs={combination[3]}, warmup_ratio={combination[4]}"
        ] = eval_result

        if eval_result["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_result["eval_loss"]
            best_params = {
                "learning_rate": combination[0],
                "per_device_train_batch_size": combination[1],
                "gradient_accumulation_steps": combination[2],
                "num_train_epochs": combination[3],
                "warmup_ratio": combination[4],
            }
            best_trainer = trained_model

        best_trainer.save_model(os.path.join(CACHE_DIR, "best_model"))

    return results, best_params

def upload_huggingface_model():
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "model.safetensors"),
        path_in_repo="model.safetensors",
        repo_id="ana-contrasto-ai/ai-wise-council",
    )
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "config.json"),
        path_in_repo="config.json",
        repo_id="ana-contrasto-ai/ai-wise-council",
    )
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "special_tokens_map.json"),
        path_in_repo="special_tokens_map.json", 
        repo_id="ana-contrasto-ai/ai-wise-council",
    )
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "tokenizer_config.json"),
        path_in_repo="tokenizer_config.json",
        repo_id="ana-contrasto-ai/ai-wise-council", 
    )
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "tokenizer.json"),
        path_in_repo="tokenizer.json",
        repo_id="ana-contrasto-ai/ai-wise-council", 
    )
    upload_file(
        path_or_fileobj=os.path.join(CACHE_DIR, "best_model", "vocab.txt"),
        path_in_repo="vocab.txt",
        repo_id="ana-contrasto-ai/ai-wise-council", 
    )

if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max_combinations", type=int, default=4)
    parser.add_argument("--dry_run", type=bool, default=False)

    args = parser.parse_args()

    results, best_params = run_finetuning(args.n_max_combinations)
    if not args.dry_run:
        upload_huggingface_model()
