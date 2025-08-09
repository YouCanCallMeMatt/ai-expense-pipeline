-- This script defines the database schema for the AI Expense Analyzer.
-- Run this file once after setting up the PostgreSQL container to create the necessary tables.

CREATE TABLE expenses (
    id SERIAL PRIMARY KEY,
    expense_id VARCHAR(10) UNIQUE NOT NULL,
    description TEXT,
    category VARCHAR(50),
    amount NUMERIC(10, 2),
    date DATE
);
