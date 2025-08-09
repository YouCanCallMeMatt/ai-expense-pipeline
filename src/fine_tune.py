import os
import time

from peft import LoraConfig, get_peft_model

from data_processing import get_expense_data
from model_utils import load_model_and_tokenizer


def main():
    """
    Main script to run the fine-tuning process.
    """
    print("--- [STAGE 2] Starting Model Fine-Tuning Process ---")

    # --- Configuration ---
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    adapter_save_path = "models/lora_adapters/expense_analyzer"

    # --- 1. Load Data ---
    # This simulates the feature extraction stage. In a real scenario,
    # this data would be pre-processed into a format suitable for training.
    expense_data = get_expense_data()
    print(f"Loaded {len(expense_data)} records for fine-tuning.")

    # --- 2. Load Model ---
    # We load the model here to apply the LoRA configuration.
    # model, _, _ = load_model_and_tokenizer(base_model_id)

    # --- 3. Apply LoRA for Efficient Fine-Tuning ---
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,
    #     task_type="CAUSAL_LM"
    # )
    # model = get_peft_model(model, lora_config)
    print("Applied LoRA adapter configuration for efficient training.")

    # --- 4. Run Training (Simulated) ---
    # In a real project, you would use the Hugging Face Trainer API here
    # with your prepared dataset to run the actual training loop.
    print("\nSimulating fine-tuning process... (This would take hours in reality)")
    time.sleep(10)

    # --- 5. Save the Trained Adapter ---
    # os.makedirs(os.path.dirname(adapter_save_path), exist_ok=True)
    # model.save_pretrained(adapter_save_path)
    print(
        f"\n✅ Fine-tuning simulation complete. LoRA adapter saved to '{adapter_save_path}'"
    )


if __name__ == "__main__":
    main()

# ==============================================================================
#  Above is a simulated fine-tuning process.
#  In a real scenario, you would load the model, apply LoRA,
#  Below is the code with the actual fine-tuning logic
#  With real model loading and saving
# ==============================================================================


# import os
# import time

# import torch
# from peft import LoraConfig, get_peft_model

# from data_processing import get_expense_data
# from model_utils import load_model_and_tokenizer


# def main():
#     """
#     Main script to run the fine-tuning process.
#     This script will load a real model, prepare it for fine-tuning,
#     and save the resulting adapter.
#     """
#     print("--- [STAGE 2] Starting Model Fine-Tuning Process ---")

#     # --- Configuration ---
#     base_model_id = "meta-llama/Meta-Llama-3-8B"
#     adapter_save_path = "models/lora_adapters/expense_analyzer"

#     # --- 1. Load Data ---
#     # This simulates the feature extraction stage. In a real scenario,
#     # this data would be pre-processed into a format suitable for training.
#     expense_data = get_expense_data()
#     print(f"Loaded {len(expense_data)} records for fine-tuning.")

#     # --- 2. Load Model ---
#     # This is now active and will download the model from Hugging Face.
#     try:
#         model, _, _ = load_model_and_tokenizer(base_model_id)
#     except Exception as e:
#         print(f"\n❌ Error loading model: {e}")
#         print(
#             "Please ensure you have accepted the Llama 3 license on Hugging Face and are logged in correctly."
#         )
#         return

#     # --- 3. Apply LoRA for Efficient Fine-Tuning ---
#     # This configures the model to only train a small number of new weights.
#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#         ],  # Modules specific to Llama models
#         lora_dropout=0.05,
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, lora_config)
#     print("\nApplied LoRA adapter configuration for efficient training.")
#     model.print_trainable_parameters()  # Shows how few parameters are being trained

#     # --- 4. Run Training (Simulated) ---
#     # In a real project, you would use the Hugging Face Trainer API here
#     # with your prepared dataset to run the actual training loop.
#     print("\nSimulating fine-tuning process... (This would take hours in reality)")
#     time.sleep(10)

#     # --- 5. Save the Trained Adapter ---
#     # This saves only the small adapter (~a few dozen MB), not the entire 8B model.
#     print(f"\nSaving trained LoRA adapter to '{adapter_save_path}'...")
#     os.makedirs(os.path.dirname(adapter_save_path), exist_ok=True)
#     model.save_pretrained(adapter_save_path)

#     print(f"\n✅ Fine-tuning simulation complete. LoRA adapter saved successfully.")


# if __name__ == "__main__":
#     main()
