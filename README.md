## AI Expense Analyzer 🤖

This project is a hardware-agnostic, microservices-based AI system designed to analyze expense data. It uses a fine-tuned Large Language Model (LLM) to understand natural language queries and provide intelligent insights about a given expense dataset.

# ✨ Features

-   **Natural Language Interaction**: Ask questions about your expense data in plain English.

-   **Intelligent Analysis**: The AI can perform tasks like listing expenses, finding the most expensive items, and identifying recent transactions.

-   **Scalable Architecture**: Built on a microservices design, allowing each component to be scaled independently.

-   **Hardware-Agnostic**: Optimized to run on a variety of hardware, not just high-end NVIDIA GPUs, thanks to modern techniques like quantization.

# 🏗️ Architecture Overview

The system is designed with a modern, scalable architecture:

-   **Microservices**: The application is broken down into independent services for Data Processing, Model Fine-Tuning, and Inference.

-   **Supervisor/Worker Model**: Each service uses a multi-agent design where a supervisor orchestrates tasks carried out by specialized worker agents in parallel.

-   **Efficient AI**: The core AI logic is built with PyTorch and optimized with techniques like 4-bit quantization (`bitsandbytes`) and LoRA (`peft`) to make large models runnable on accessible hardware.

# 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

-   Python 3.10 or newer

-   Conda for environment management

-   A Hugging Face account with approved access to a Llama 3 model.

-   A machine with an NVIDIA GPU is highly recommended for reasonable performance.

### 2. Project Setup

First, clone the repository and set up the Conda environment.

```
# Clone the project repository (if applicable)
# git clone <your-repo-url>
# cd ai_expense_project

# Create and activate a new conda environment
conda create -n expense-ai python=3.10
conda activate expense-ai

```

### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```
pip install -r requirements.txt

```

### 4. Hugging Face Authentication

Log in to your Hugging Face account from your terminal. This is required to download the pre-trained Llama 3 model.

```
huggingface-cli login

```

You will be prompted to paste an access token from your Hugging Face account.

# Usage

The project is structured into separate scripts within the `src/` directory.

### 1. Fine-Tune the Model (Run Once)

Before you can run the main application, you need to fine-tune the base LLM on your specific expense data. This script simulates that process and creates a LoRA adapter.

```
python src/fine_tune.py

```

This will create a new adapter in the `models/lora_adapters/` directory. In a real-world scenario, this step would involve a proper training dataset and might take a significant amount of time.

### 2. Run the Interactive Application

Once the model adapter is ready, you can start the main application to chat with your AI.

```
python src/app.py

```

This will load the base Llama 3 model, apply your fine-tuned adapter, and open an interactive prompt in your terminal. You can then start asking questions about the expense data.

**Example Prompts:**

-   `List me the top 5 expenses`

-   `what is the most recent expense?`

-   `show me 3 latest expenses`
