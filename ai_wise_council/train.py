import os
import re
import string
from typing import Tuple

import evaluate
import numpy as np
import pandas as pd
import bitsandbytes as bnb
import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding

# import wandb
from trl import SFTTrainer

from ai_wise_council import CACHE_DIR

load_dotenv()
# wandb.login(key=os.getenv("WAB_KEY"), verify=True)


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


# https://www.datacamp.com/tutorial/fine-tuning-phi-3-5
# https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py
model_id = "microsoft/Phi-3-mini-4k-instruct"
model_name = "microsoft/Phi-3-mini-4k-instruct"
new_model = "ai-wise-council"
hf_model_repo = "UserName/" + new_model
# LoRA parameters
lora_alpha = 16
lora_dropout = 0
lora_r = 64
target_modules = [
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="float16",
        quantization_config=bnb_config, 
        cache_dir=CACHE_DIR,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_dataset() -> pd.DataFrame:
    """
    Load the debate dataset from CSV file.
    The dataset contains debate conversations between AI agents, where one agent argues in good faith
    and another argues in bad faith. The dataset is stored in data/output/debate_dataset.csv
    relative to the project root.

    Returns:
        pd.DataFrame: DataFrame containing the debate conversations and metadata
    """
    from pathlib import Path

    df = pd.read_csv(
        Path(__file__).parents[1] / "data/output/debate_dataset.csv"
    )
    return df


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
        debater_ids = re.findall(r"<DEBATER ID: (\d+)>", text)
        unique_ids = list(dict.fromkeys(debater_ids))  # preserve order of appearance

        result = text
        id_debater_good_faith_new = ""
        for i, id in enumerate(unique_ids):
            upper_str = string.ascii_uppercase[i]
            result = result.replace(f"<DEBATER ID: {id}>", f"<DEBATER ID: {upper_str}>")
            result = result.replace(
                f"</DEBATER ID: {id}>", f"<DEBATER ID: {upper_str}>"
            )
            if str(id) == str(id_debater_good_faith):
                id_debater_good_faith_new = i
        return result, id_debater_good_faith_new

    df_train = df.sample(frac=train_size, random_state=42)
    df_test = df.drop(df_train.index)

    # Replace the loop with apply
    df_train[["debate", "id_debater_good_faith"]] = df_train.apply(
        lambda x: _replace_debater_ids(x.debate, x.id_debater_good_faith),
        axis=1,
        result_type="expand",
    )

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # print(next(model.parameters()).device)

    # wandb.init(project=new_model, name = "phi-3-mini-ft-py-3e")

    if training_args is None:
        # default parameters
        training_args = {
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 3,
            "warmup_ratio": 0.1,
        }

    output_dir = "./phi-3-mini-LoRA"
    if CACHE_DIR:
        output_dir = os.path.join(CACHE_DIR, "phi-3-mini-LoRA")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args["num_train_epochs"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args[
            "gradient_accumulation_steps"
        ],  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="paged_adamw_8bit",
        logging_steps=100,
        learning_rate=training_args["learning_rate"],
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        max_steps=-1,
        warmup_ratio=training_args["warmup_ratio"],
        evaluation_strategy="steps",
        lr_scheduler_type="cosine",  # use cosine learning rate scheduler
        eval_strategy="steps",  # save checkpoint every epoch
        eval_steps=0.2,
        # do_eval=True,
        # log_level="debug",
        # save_strategy="epoch",
        # report_to="wandb",
    )

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )


def run_training(training_args):
    import logging

    logging.basicConfig(level=logging.INFO)

    print(f"CACHE_DIR: {CACHE_DIR}")
    print("Loading dataset...")
    df = load_dataset()
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name)
    print("Preparing and splitting dataset encodings...")
    train_encodings, test_encodings, df_train, df_test = prepare_datasets(df, tokenizer)
    print(f"Split into {len(df_train)} training and {len(df_test)} test examples")

    print("Creating dataset objects...")
    train_dataset = PromptDataset(
        train_encodings, df_train["id_debater_good_faith"].tolist()
    )
    test_dataset = PromptDataset(
        test_encodings, df_test["id_debater_good_faith"].tolist()
    )
    print("Dataset objects created successfully")

    print("Initializing trainer...")
    trainer = create_trainer(model, train_dataset, test_dataset, training_args)

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    return trainer

    # print("Saving model...")
    # trainer.save_model("final_model")
    # print("Training completed and model saved to 'final_model' directory!")


if __name__ == "__main__":
    training_args = None
    run_training(training_args)
