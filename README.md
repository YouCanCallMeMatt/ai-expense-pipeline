
## AI Expense Analyzer 🤖

This project is a hardware-agnostic, microservices-based AI system designed to analyze expense data. It uses a fine-tuned Large Language Model (LLM) to understand natural language queries and provide intelligent insights about a given expense dataset.

# ✨ Features

* **Natural Language Interaction**: Ask questions about your expense data in plain English.
* **Intelligent Analysis**: The AI can perform tasks like listing expenses, finding the most expensive items, and identifying recent transactions.
* **Scalable Architecture**: Built on a microservices design, allowing each component to be scaled independently.
* **Hardware-Agnostic**: Optimized to run on a variety of hardware, not just high-end NVIDIA GPUs, thanks to modern techniques like quantization.

# 🏗️ Architecture Overview

The system is designed with a modern, scalable architecture:

* **Microservices**: The application is broken down into independent services for Data Processing, Model Fine-Tuning, and Inference.
* **Supervisor/Worker Model**: Each service uses a multi-agent design where a supervisor orchestrates tasks carried out by specialized worker agents in parallel.
* **Efficient AI**: The core AI logic is built with PyTorch and optimized with techniques like 4-bit quantization (`bitsandbytes`) and LoRA (`peft`) to make large models runnable on accessible hardware.

# 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.10 or newer
* Conda for environment management
* Docker and Docker Compose
* A Hugging Face account with approved access to a Llama 3 model.
* A machine with an NVIDIA GPU is highly recommended for reasonable performance.

### 2. Project Setup

First, clone the repository and set up the Conda environment.

```bash
# Clone the project repository (if applicable)
# git clone <your-repo-url>
# cd ai_expense_project

# Create and activate a new conda environment
conda create -n expense-ai python=3.10
conda activate expense-ai
```

### 3. Set Up PostgreSQL Database

This project uses a PostgreSQL database running in Docker.

```bash
# Start the database container in the background
docker-compose up -d
```
This command will start a PostgreSQL 14 instance based on the `docker-compose.yml` file in the project root.

### 4. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

This project uses a `.env` file to manage configuration and secrets.

```bash
# Copy the example file to create your own local configuration
cp .env.example .env
```
Now, open the newly created `.env` file and fill in your specific details (like your database password).

### 6. Create Database Table

Run the `schema.sql` script to create the `expenses` table in your database. You will be prompted for your database password.

```bash
psql -h localhost -d postgres -U postgres -f schema.sql
```

### 7. Hugging Face Authentication

Log in to your Hugging Face account from your terminal. This is required to download the pre-trained Llama 3 model.

```bash
huggingface-cli login
```
You will be prompted to paste an access token from your Hugging Face account.

# Usage

The project is structured into separate scripts within the `src/` directory.

### 1. Populate the Database (Run Once)

Before running the main application, you need to populate your database with the sample expense data.

```bash
python src/populate_db.py
```

### 2. Verifying the Database Setup (Optional)

You can manually check the database to ensure the table and data were created correctly.

**a. Connect to the Database:**
Use `psql` to connect. You will be prompted for your password.
```bash
psql -h localhost -d postgres -U postgres
```

**b. Check if the Table Exists:**
Once connected, run the `\dt` command to list all tables.
```sql
\dt
```
You should see the `expenses` table in the list.

**c. Check the Table Content:**
Run a `SELECT` query to see the first 5 rows of data.
```sql
SELECT * FROM expenses LIMIT 5;
```
This confirms that the `populate_db.py` script ran successfully. Type `\q` to exit `psql`.

### 3. Fine-Tune the Model (Run Once)

Next, run the fine-tuning script to train the LoRA adapter on your specific data. This step is simulated for speed.

```bash
python src/fine_tune.py
```
This will create a new adapter in the `models/lora_adapters/` directory.

### 4. Run the Interactive Application

Once the setup is complete, you can start the main application to chat with your AI.

```bash
python src/app.py
```
This will load the base Llama 3 model, apply your fine-tuned adapter, and open an interactive prompt in your terminal.

**Example Prompts:**

* `List me the top 5 expenses`
* `what is the most recent expense?`
* `show me 3 latest expenses`
```