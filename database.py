from modules.lib import *

# Function to initialize the database with separate tables
def sep_initialize_database():
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()

    # Create table for Graph Data  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Graph_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            date TEXT,
            ndvi_value REAL,
            report_type TEXT 
        )
    ''')

    # Create table for Histogram Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Histogram_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            data REAL,
            report_type TEXT 
        )
    ''')
    # Create table for Heatmap Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Heatmap_Reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            ndvi_value REAL,
            report_type TEXT 
        )
    ''')
    # Create table for Surface Plot Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Surface_data_report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            ndvi_value REAL,
            report_type TEXT 
        )
    ''')
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS User_Feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Call this function when starting the app to ensure tables exist
sep_initialize_database()


def clear_database():
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()

    # Fetch all table names dynamically
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"DELETE FROM {table}") 

    conn.commit()
    conn.close()
    st.success("✅ All database tables have been cleared!")

# Register cleanup function when Streamlit stops
atexit.register(clear_database)























# # Function to fetch all saved reports
# def fetch_saved_reports():
#     conn = sqlite3.connect("ndvi_reports.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, report_name, report_type FROM reports")
#     reports = cursor.fetchall()
#     conn.close()
#     return reports

# # Function to fetch a specific report
# def get_report_data(report_id):
#     conn = sqlite3.connect("ndvi_reports.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT data FROM reports WHERE id=?", (report_id,))
#     data = cursor.fetchone()
#     conn.close()
#     if data:
#         return pd.read_csv(StringIO(data[0]))
#     return None
