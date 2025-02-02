from itertools import product
import random

import torch

from ai_wise_council.train import run_training

PARAM_GRID = {
    'learning_rate': [1e-5, 5e-5],
    'per_device_train_batch_size': [4, 8],
    'per_gpu_batch_size': [16, 32, 64],
    'num_train_epochs': [2, 3, 4],
    'warmup_ratio': [0.3, 0.01, 0.001],
}

def run_finetuning(n_max_combinations: int):
    param_combinations = list(product(
        PARAM_GRID['learning_rate'],
        PARAM_GRID['per_device_train_batch_size'],
        PARAM_GRID['per_gpu_batch_size'],
        PARAM_GRID['num_train_epochs'],
        PARAM_GRID['warmup_ratio']
    ))

    if len(param_combinations) > n_max_combinations:
        param_combinations = random.sample(param_combinations, n_max_combinations)

    print(f"Running {len(param_combinations)} combinations")

    # run combinations

    results = {}
    best_eval_loss = float('inf')

    for combination in param_combinations:
        training_args = {
            'learning_rate': combination[0],
            'per_device_train_batch_size': combination[1],
            'per_device_eval_batch_size': combination[2],
            'num_train_epochs': combination[3],
            'warmup_ratio': combination[4],
        }
        torch.cuda.memory_summary(device=None, abbreviated=False)
        trained_model = run_training(training_args)
        eval_result = trained_model.evaluate()

        results[f"lr={combination[0]}, per_device_train_batch_size={combination[1]}, per_device_eval_batch_size={combination[2]}, num_train_epochs={combination[3]}, warmup_ratio={combination[4]}"] = eval_result
        
        if eval_result['eval_loss'] < best_eval_loss:
            best_eval_loss = eval_result['eval_loss']
            best_params = {
                'learning_rate': combination[0],
                'per_device_train_batch_size': combination[1],
                'per_device_eval_batch_size': combination[2],
                'num_train_epochs': combination[3],
                'warmup_ratio': combination[4],
            }
            best_trainer = trained_model

        best_trainer.save_model("best_model")

    return results, best_params


if __name__ == "__main__":
    run_finetuning(2)



