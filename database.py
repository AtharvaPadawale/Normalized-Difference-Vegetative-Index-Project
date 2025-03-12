import sqlite3
import pandas as pd
from io import StringIO

# Function to save report data to the database
def save_report_to_db(report_name, data, report_type):
    conn = sqlite3.connect("ndvi_reports.db")
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            report_type TEXT,
            data TEXT
        )
    ''')

    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Insert report into the database
    cursor.execute("INSERT INTO reports (report_name, report_type, data) VALUES (?, ?, ?)", 
                   (report_name, report_type, csv_content))
    
    conn.commit()
    conn.close()

# Function to fetch all saved reports
def fetch_saved_reports():
    conn = sqlite3.connect("ndvi_reports.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, report_name, report_type FROM reports")
    reports = cursor.fetchall()
    conn.close()
    return reports

# Function to fetch a specific report
def get_report_data(report_id):
    conn = sqlite3.connect("ndvi_reports.db")
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM reports WHERE id=?", (report_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        return pd.read_csv(StringIO(data[0]))
    return None
