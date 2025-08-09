import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_agnostic_device():
    """Selects the best available hardware device."""
    if torch.cuda.is_available():
        print("✅ CUDA (NVIDIA GPU) detected. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) detected. Using GPU.")
        return torch.device("mps")
    else:
        print(
            "⚠️ Warning: No GPU detected. Falling back to CPU. This will be very slow."
        )
        return torch.device("cpu")


def load_model_and_tokenizer(base_model_id, adapter_path=None):
    """
    Loads the quantized base model and tokenizer from Hugging Face.
    Optionally applies a fine-tuned LoRA adapter.
    """
    device = get_agnostic_device()

    print(f"\n--- Loading model: {base_model_id} ---")
    print(
        "This may take several minutes on the first run as the model is downloaded..."
    )

    # Configure quantization to load the model with less memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the quantized base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses GPU if available
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to suppress warnings

    # If an adapter path is provided, load the fine-tuned adapter
    if adapter_path:
        print(f"Applying fine-tuned adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    print("\n✅ Model and tokenizer loaded successfully!")
    return model, tokenizer, device
