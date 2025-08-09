import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables from the .env file in the project root
load_dotenv()

# --- Database Connection Details (Now loaded from .env) ---
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to the PostgreSQL database: {e}")
        print(
            "Please ensure PostgreSQL is running and the .env file details are correct."
        )
        return None


def get_expense_data():
    """
    Loads expense data by querying the PostgreSQL database.
    """
    print("Connecting to database to fetch expense data...")
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT expense_id, description, category, amount, date FROM expenses ORDER BY date DESC"
            )
            expenses = cur.fetchall()
        print(f"✅ Successfully loaded {len(expenses)} records from the database.")
        return expenses
    except Exception as e:
        print(f"❌ Error fetching data from the database: {e}")
        return []
    finally:
        if conn:
            conn.close()
