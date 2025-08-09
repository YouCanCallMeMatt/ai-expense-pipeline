import time
from datetime import datetime

import torch

from data_processing import get_expense_data
from model_utils import load_model_and_tokenizer


def main():
    """
    Main application to run interactive inference with the fine-tuned model.
    """
    # --- Configuration ---
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    adapter_path = (
        "models/lora_adapters/expense_analyzer"  # Path to our fine-tuned adapter
    )

    # --- 1. Load Data for RAG/Context ---
    # The AI will use this data to answer questions.
    expense_data = get_expense_data()

    # --- 2. Load the Model and Tokenizer ---
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            base_model_id, adapter_path=None
        )  # Set adapter_path to run with it
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(
            "Please ensure you have accepted the Llama 3 license on Hugging Face and are logged in correctly."
        )
        return

    # --- 3. Start Interactive Prompt Session ---
    print("\n--- STARTING INTERACTIVE EXPENSE ANALYZER ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_prompt = input("Your prompt > ")
            if user_prompt.lower() in ["quit", "exit"]:
                print("Exiting interactive session.")
                break

            # --- This section simulates RAG ---
            # A real RAG system would first search the data for relevant context.
            # Here, we provide the full dataset as context for simplicity.
            context = "You are an AI Expense Analyzer. Your task is to answer the user's question based on the provided expense data. Be concise.\n\n"
            context += (
                "## Expense Data:\n" + str(expense_data[:5]) + "\n... (and more)\n\n"
            )  # Add a snippet of data as context

            full_prompt = context + f"## User Query:\n{user_prompt}"

            # Tokenize the input
            model_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            # Generate a response
            print("AI is thinking...")
            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end_time = time.time()

            # Decode the response
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Clean up the prediction to remove the initial prompt
            # Find where the user query ends and take the text after it
            answer_start_index = prediction.find(user_prompt) + len(user_prompt)
            final_answer = prediction[answer_start_index:].strip()

            print(
                f"\nAI Response (generated in {end_time - start_time:.2f}s):\n{final_answer}\n"
            )

        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    main()
