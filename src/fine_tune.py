
import os
from dotenv import load_dotenv
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from model_utils import load_model_and_tokenizer
from data_processing import get_expense_data

load_dotenv()

def format_prompt(sample):
    """Formats a data sample into a prompt for fine-tuning."""
    instruction = f"Analyze the following expense: {sample['description']} in category {sample['category']} with amount ${sample['amount']}."
    response = f"This is a {sample['category']} expense for {sample['description']} costing ${sample['amount']}."
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": prompt}

def main():
    """
    Main script to run the fine-tuning process.
    It now checks if an adapter already exists and asks the user before re-running.
    """
    print("--- [STAGE 2] Starting Model Fine-Tuning Process ---")
    
    # --- Configuration (loaded from .env) ---
    base_model_id = os.getenv("BASE_MODEL_ID")
    lora_modules_str = os.getenv("LORA_TARGET_MODULES")
    
    # --- Create model-specific paths ---
    adapter_save_path = f"models/lora_adapters/{base_model_id.replace('/', '_')}"
    training_output_dir = f"./training_output/{base_model_id.replace('/', '_')}"

    # --- Check if the adapter already exists and ask the user ---
    if os.path.exists(adapter_save_path):
        print(f"✅ Fine-tuned adapter already exists for '{base_model_id}' at '{adapter_save_path}'.")
        user_input = input("Do you want to run fine-tuning again? This will overwrite the existing adapter. (yes/no): ")
        if user_input.lower() not in ['yes', 'y']:
            print("Skipping fine-tuning.")
            return

    if not lora_modules_str:
        print("❌ Error: LORA_TARGET_MODULES not found in .env file. Please define it.")
        return
        
    lora_target_modules = [module.strip() for module in lora_modules_str.split(',')]

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
    formatted_dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    print("\nData prepared and formatted for training.")

    # --- 3. Apply LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Set Up and Run Training ---
    training_args = TrainingArguments(
        output_dir=training_output_dir, # Use the model-specific output directory
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_dir=f'./logs/{base_model_id.replace("/", "_")}',
        logging_steps=10,
        learning_rate=2e-4,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
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