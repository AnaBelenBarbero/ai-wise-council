from typing import Tuple
import os
import re
import string

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding
from peft import LoraConfig, TaskType
import evaluate
import wandb
from trl import SFTTrainer


load_dotenv()
wandb.login(key=os.getenv("WAB_KEY"), verify=True)


DEVICE_MAP = {"": 0}


class PromptDataset(Dataset):
    """Dataset class for prompt classification"""

    def __init__(self, encodings, labels: int):
        self.encodings: BatchEncoding | dict = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py
model_id = "microsoft/Phi-3-mini-4k-instruct"
model_name = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
dataset_split= "train"
new_model = "ai-wise-council"
hf_model_repo="UserName/"+new_model
# LoRA parameters
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]


def setup_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Setup the model and tokenizer

    Args:
        model_name: Name or path of the model

    Returns:
        Tuple of (model, tokenizer)
    """
    if torch.cuda.is_bf16_supported() and False: # TODO: UNMOCK
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, use_auth_token=True, 
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,
        device_map=DEVICE_MAP,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return model, tokenizer

def load_dataset() -> pd.DataFrame:
    """
    Load and combine all datasets

    Args:
        english_path: Path to English dataset
        spanish_jailbreak_path: Path to Spanish jailbreak dataset
        spanish_benign_path: Path to Spanish benign dataset

    Returns:
        Combined DataFrame
    """
    from pathlib import Path

    # english
    df_3k_neg = pd.read_csv(Path(__file__).parents[1] / "notebooks/data/output/debate_dataset.csv")
    return df_3k_neg


def prepare_datasets(
    df: pd.DataFrame, tokenizer: AutoTokenizer, train_size: float = 0.8
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Prepare and split datasets

    Args:
        df: Input DataFrame
        tokenizer: HuggingFace tokenizer
        train_size: Fraction of data for training

    Returns:
        Tuple of (train_encodings, test_encodings, train_df, test_df)
    """
    def _replace_debater_ids(text: str, id_debater_good_faith: str) -> str:
        """
        Replace debater IDs with uppercase letters (A, B, C, etc.) while preserving order

        Args:
            text: Input text containing debater ID tags

        Returns:
            Text with debater IDs replaced by uppercase letters
        """
        debater_ids = re.findall(r'<DEBATER ID: (\d+)>', text)
        unique_ids = list(dict.fromkeys(debater_ids))  # preserve order of appearance

        result = text
        id_debater_good_faith_new = ''
        for i, id in enumerate(unique_ids):
            upper_str = string.ascii_uppercase[i]
            result = result.replace(f"<DEBATER ID: {id}>", f"<DEBATER ID: {upper_str}>")
            result = result.replace(f"</DEBATER ID: {id}>", f"<DEBATER ID: {upper_str}>")
            if str(id) == str(id_debater_good_faith):
                id_debater_good_faith_new = i
        return result, id_debater_good_faith_new
    
    df_train = df.sample(frac=train_size, random_state=42)
    df_test = df.drop(df_train.index)
    
    # Replace the loop with apply
    df_train[["debate", "id_debater_good_faith"]] = df_train.apply(lambda x: _replace_debater_ids(x.debate, x.id_debater_good_faith), axis=1, result_type='expand')

    # tokenize
    train_encodings: BatchEncoding | dict = tokenizer(
        df_train["debate"].tolist(),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    test_encodings: BatchEncoding | dict = tokenizer(
        df_test["debate"].tolist(),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # convert to numpy for dataset creation
    train_encodings = {key: val.numpy() for key, val in train_encodings.items()}
    test_encodings = {key: val.numpy() for key, val in test_encodings.items()}

    return train_encodings, test_encodings, df_train, df_test


def compute_metrics(eval_pred) -> dict:
    """
    Compute metrics for model evaluation

    Args:
        eval_pred: Tuple of predictions and labels

    Returns:
        Dictionary of metric scores
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def create_trainer(
    model: AutoModelForSequenceClassification,
    train_dataset: PromptDataset,
    eval_dataset: PromptDataset,
    training_args: dict | None = None,
) -> Trainer:
    """
    Create and return a trainer instance

    Args:
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Optional training arguments

    Returns:
        Trainer instance
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    #print(next(model.parameters()).device)

    wandb.init(project=new_model, name = "phi-3-mini-ft-py-3e")

    if training_args is None:
        args = TrainingArguments(
            output_dir="./phi-3-mini-LoRA",
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            log_level="debug",
            save_strategy="epoch",
            logging_steps=100,
            learning_rate=1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=100,
            num_train_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            report_to="wandb",
            seed=42,
        )

        peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
        )

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=args,
    )

if __name__ == "__main__":
    df = load_dataset()
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    print("Preparing and splitting dataset encodings...")
    train_encodings, test_encodings, df_train, df_test = prepare_datasets(df, tokenizer)
    print(f"Split into {len(df_train)} training and {len(df_test)} test examples")

    print("Creating dataset objects...")
    train_dataset = PromptDataset(train_encodings, df_train["id_debater_good_faith"].tolist())
    test_dataset = PromptDataset(test_encodings, df_test["id_debater_good_faith"].tolist())
    print("Dataset objects created successfully")

    print("Initializing trainer...")
    trainer = create_trainer(model, train_dataset, test_dataset)

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    print("Saving model...")
    trainer.save_model("final_model")
    print("Training completed and model saved to 'final_model' directory!")