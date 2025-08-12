import os
import time

import torch
from dotenv import load_dotenv

from data_processing import get_expense_data
from model_utils import load_model_and_tokenizer

# Load environment variables from the .env file
load_dotenv()


def main():
    """
    Main application to run interactive inference with the fine-tuned model.
    This script loads the model and enters a loop to chat with the user.
    """
    # --- Configuration (Now loaded from .env) ---
    base_model_id = os.getenv("BASE_MODEL_ID")
    # The path to the adapter is dynamically created based on the model name.
    adapter_path = f"models/lora_adapters/{base_model_id.replace('/', '_')}"

    # --- 1. Load Data for RAG/Context ---
    # The AI will use this data to answer questions.
    expense_data = get_expense_data()

    # --- 2. Load the Model and Tokenizer ---
    try:
        # Load the base model and apply the fine-tuned adapter on top.
        model, tokenizer, device = load_model_and_tokenizer(
            base_model_id, adapter_path=adapter_path
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(
            "Please ensure you have accepted the model's license on Hugging Face and are logged in correctly."
        )
        return

    print("\n--- STARTING INTERACTIVE EXPENSE ANALYZER ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    # --- 3. Start Interactive Prompt Session ---
    while True:
        try:
            user_prompt = input("Your prompt > ")
            if user_prompt.lower() in ["quit", "exit"]:
                print("Exiting interactive session.")
                break

            # --- UPDATED: More effective RAG prompt template ---
            # This template is more direct and helps override the model's default safety behavior.
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that analyzes expense data. You must answer the user's question based *only* on the context provided below. Do not use any outside knowledge. If the answer is not in the context, say that you cannot find the information.",
                },
                {
                    "role": "user",
                    "content": f"""
                    Context:
                    Here is the expense data:
                    {str(expense_data[:20])}
                    ---
                    Question:
                    {user_prompt}
                    """,
                },
            ]

            # This creates the full prompt string ready for the model.
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Convert the text prompt into numbers (tokens) that the model can process.
            model_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            print("AI is thinking...")
            start_time = time.time()
            with torch.no_grad():  # Disables gradient calculations to save memory and speed up inference.
                # The model generates a sequence of new tokens based on the input.
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=250,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end_time = time.time()

            # Decode only the newly generated tokens to get the clean answer, ignoring the input prompt.
            response_ids = output_ids[0][model_inputs.input_ids.shape[1] :]
            final_answer = tokenizer.decode(response_ids, skip_special_tokens=True)

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
