import random
from datetime import datetime, timedelta

import psycopg2

from data_processing import (
    get_db_connection,
)  # Reuse the connection logic from data_processing.py


def generate_fake_expenses(num_records=50):
    """
    Generates a list of realistic fake expense records for testing purposes.
    This creates a diverse dataset with different categories, descriptions, and amounts.
    """
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
        # Each expense is created as a tuple, which is an efficient format for batch database insertion.
        expense = (
            f"E{1001 + i}",
            random.choice(descriptions[cat]),
            cat,
            round(random.uniform(25.50, 500.75), 2),
            (datetime.now() - timedelta(days=random.randint(1, 365))).date(),
        )
        expenses.append(expense)
    return expenses


def main():
    """
    Connects to the database and inserts 50 fake expense records.
    This script should be run once to populate the database with initial data.
    """
    print("--- Starting Database Population Script ---")
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            # First, check if the table already has data to avoid inserting duplicates.
            cur.execute("SELECT COUNT(*) FROM expenses")
            if cur.fetchone()[0] > 0:
                print("⚠️ Database already contains data. Skipping population.")
                return

            # If the table is empty, generate and insert new data.
            print("Generating 50 fake expense records...")
            expenses_to_insert = generate_fake_expenses(50)

            insert_query = "INSERT INTO expenses (expense_id, description, category, amount, date) VALUES (%s, %s, %s, %s, %s)"

            # Use `execute_batch` for efficient insertion of many rows at once.
            psycopg2.extras.execute_batch(cur, insert_query, expenses_to_insert)

            # Commit the transaction to make the changes permanent in the database.
            conn.commit()

            print(
                f"✅ Successfully inserted {len(expenses_to_insert)} records into the 'expenses' table."
            )

    except Exception as e:
        print(f"❌ An error occurred during database population: {e}")
    finally:
        # Always close the connection, whether the process succeeds or fails.
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
