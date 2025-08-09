import os
import random
from datetime import datetime, timedelta

import pandas as pd


def generate_fake_expenses(num_records=50):
    """Generates a list of realistic fake expense records."""
    categories = ["Travel", "Office Supplies", "Software", "Food", "Utilities"]
    descriptions = {
        "Travel": [
            "Flight to client meeting",
            "Hotel for conference",
            "Taxi from airport",
        ],
        "Office Supplies": [
            "Printer paper and ink",
            "New keyboards",
            "Whiteboard markers",
        ],
        "Software": [
            "Monthly subscription for CRM",
            "Annual license for IDE",
            "Cloud hosting bill",
        ],
        "Food": ["Team lunch meeting", "Coffee for the office", "Catering for event"],
        "Utilities": ["Monthly electricity bill", "Internet service provider bill"],
    }
    expenses = []
    for i in range(num_records):
        cat = random.choice(categories)
        expense = {
            "expense_id": f"E{1001 + i}",
            "description": random.choice(descriptions[cat]),
            "category": cat,
            "amount": round(random.uniform(25.50, 500.75), 2),
            "date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime(
                "%Y-%m-%d"
            ),
        }
        expenses.append(expense)
    return expenses


def get_expense_data(file_path="data/raw/expenses.csv"):
    """
    Loads expense data from a CSV file. If the file doesn't exist,
    it generates new fake data and saves it.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        print(
            f"Data file not found. Generating new fake data and saving to '{file_path}'..."
        )
        expenses = generate_fake_expenses(50)
        df = pd.DataFrame(expenses)
        df.to_csv(file_path, index=False)
    else:
        print(f"Loading data from '{file_path}'...")
        df = pd.read_csv(file_path)

    # Convert dataframe to list of dictionaries for easier processing
    return df.to_dict("records")
