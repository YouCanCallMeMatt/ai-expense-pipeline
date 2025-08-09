import random
from datetime import datetime, timedelta

import psycopg2

from data_processing import get_db_connection  # Reuse the connection logic


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
    This script should be run once to populate the database.
    """
    print("--- Starting Database Population Script ---")
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            # Check if the table already has data
            cur.execute("SELECT COUNT(*) FROM expenses")
            if cur.fetchone()[0] > 0:
                print("⚠️ Database already contains data. Skipping population.")
                return

            print("Generating 50 fake expense records...")
            expenses_to_insert = generate_fake_expenses(50)

            insert_query = "INSERT INTO expenses (expense_id, description, category, amount, date) VALUES (%s, %s, %s, %s, %s)"

            psycopg2.extras.execute_batch(cur, insert_query, expenses_to_insert)
            conn.commit()

            print(
                f"✅ Successfully inserted {len(expenses_to_insert)} records into the 'expenses' table."
            )

    except Exception as e:
        print(f"❌ An error occurred during database population: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
