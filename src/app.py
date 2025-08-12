
import os
import torch
import time
from dotenv import load_dotenv
from model_utils import load_model_and_tokenizer
from data_processing import get_expense_data

load_dotenv()

def format_data_for_prompt(expense_data):
    """Converts the raw list of database rows into a clean, human-readable string."""
    if not expense_data:
        return "No expense data available."
    if not isinstance(expense_data[0], dict):
         expense_data = [dict(row) for row in expense_data]
    formatted_lines = []
    for expense in expense_data:
        date_str = expense['date'].strftime('%Y-%m-%d') if hasattr(expense['date'], 'strftime') else expense['date']
        line = f"- ID: {expense['expense_id']}, Description: {expense['description']}, Amount: ${expense['amount']}, Date: {date_str}"
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

def main():
    """Main application to run interactive inference with the fine-tuned model."""
    # --- Configuration (Now loaded from .env) ---
    base_model_id = os.getenv("BASE_MODEL_ID")
    # --- NEW: Dynamically create the adapter path based on the model ---
    adapter_path = f"models/lora_adapters/{base_model_id.replace('/', '_')}"
    
    expense_data = get_expense_data()

    try:
        # Load the base model and apply the correct fine-tuned adapter on top
        model, tokenizer, device = load_model_and_tokenizer(base_model_id, adapter_path=adapter_path)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please ensure you have accepted the model's license on Hugging Face and are logged in correctly.")
        return

    print("\n--- STARTING INTERACTIVE EXPENSE ANALYZER ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_prompt = input("Your prompt > ")
            if user_prompt.lower() in ['quit', 'exit']:
                print("Exiting interactive session.")
                break

            formatted_context = format_data_for_prompt(expense_data[:50])

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that analyzes expense data. You must answer the user's question based *only* on the context provided below. Summarize the findings in a clear, human-readable format. For lists, use bullet points."},
                {"role": "user", "content": f"Context:\nHere is the expense data:\n{formatted_context}\n---\nQuestion:\n{user_prompt}"}
            ]
            
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            print("AI is thinking...")
            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(**model_inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
            end_time = time.time()
            
            response_ids = output_ids[0][model_inputs.input_ids.shape[1]:]
            final_answer = tokenizer.decode(response_ids, skip_special_tokens=True)

            print(f"\nAI Response (generated in {end_time - start_time:.2f}s):\n{final_answer}\n")

        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()