import os

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from data_processing import get_expense_data
from model_utils import load_model_and_tokenizer

# Load environment variables from the .env file
load_dotenv()


def format_prompt(sample):
    """
    Formats a single data sample into a structured prompt for fine-tuning.
    This version returns a dictionary with a single 'text' field.
    """
    instruction = f"Analyze the following expense: {sample['description']} in category {sample['category']} with amount ${sample['amount']}."
    response = f"This is a {sample['category']} expense for {sample['description']} costing ${sample['amount']}."

    # This format is a simplified instruction-following template.
    # A real tokenizer's chat template would add special tokens.
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": prompt}


def main():
    """
    Main script to run the fine-tuning process. This script loads a pre-trained model,
    prepares a dataset, and then fine-tunes the model using LoRA.
    """
    print("--- [STAGE 2] Starting REAL Model Fine-Tuning Process ---")

    # --- Configuration (loaded from .env) ---
    base_model_id = os.getenv("BASE_MODEL_ID")
    lora_modules_str = os.getenv("LORA_TARGET_MODULES")
    adapter_save_path = f"models/lora_adapters/{base_model_id.replace('/', '_')}"

    if not lora_modules_str:
        print("❌ Error: LORA_TARGET_MODULES not found in .env file. Please define it.")
        return

    lora_target_modules = [module.strip() for module in lora_modules_str.split(",")]

    print(f"Using base model: {base_model_id}")
    print(f"LoRA target modules: {lora_target_modules}")

    # --- 1. Load Model and Tokenizer ---
    try:
        model, tokenizer, _ = load_model_and_tokenizer(base_model_id)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return

    # --- 2. Prepare Data ---
    expense_data = get_expense_data()
    dataset = Dataset.from_list(expense_data)

    # First, format the prompts into a single 'text' column
    formatted_dataset = dataset.map(format_prompt)

    # Second, tokenize the 'text' column to create 'input_ids' and 'attention_mask'
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

    # **FIX:** Remove the original text columns that the Trainer doesn't need.
    tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["text"]
    )  # Also remove the intermediate 'text' column

    # **FIX:** The trainer needs a 'labels' column for language modeling.
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]}, batched=True
    )

    print("\nData prepared and formatted for training.")

    # --- 3. Apply LoRA for Efficient Fine-Tuning ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Set Up and Run Training ---
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Use a smaller batch size for CPU
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=2e-4,
    )

    # Use a standard data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # --- 5. Save the Trained Adapter ---
    print(f"\nSaving trained LoRA adapter to '{adapter_save_path}'...")
    os.makedirs(os.path.dirname(adapter_save_path), exist_ok=True)
    model.save_pretrained(adapter_save_path)

    print(f"\n✅ Fine-tuning complete. LoRA adapter saved successfully.")


if __name__ == "__main__":
    main()
