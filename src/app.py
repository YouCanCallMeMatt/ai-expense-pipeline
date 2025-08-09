import os
import time

import torch
from dotenv import load_dotenv

from data_processing import get_expense_data
from model_utils import load_model_and_tokenizer

# Load environment variables
load_dotenv()


def main():
    """
    Main application to run interactive inference with the fine-tuned model.
    """
    # --- Configuration (Now loaded from .env) ---
    base_model_id = os.getenv("BASE_MODEL_ID")
    adapter_path = "models/lora_adapters/expense_analyzer"

    # ... (rest of the app logic remains the same) ...

    print(f"Loading model '{base_model_id}' for inference...")
    # For this simulation, we'll skip the actual model loading to keep it fast.
    # In the real script, this would call load_model_and_tokenizer.

    print("\n--- STARTING INTERACTIVE EXPENSE ANALYZER (SIMULATION) ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_prompt = input("Your prompt > ")
            if user_prompt.lower() in ["quit", "exit"]:
                print("Exiting interactive session.")
                break

            print("AI is thinking...")
            time.sleep(1)
            print(f"AI Response: This is a simulated response for '{user_prompt}'.\n")

        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break


if __name__ == "__main__":
    main()
