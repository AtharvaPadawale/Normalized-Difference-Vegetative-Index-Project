from modules.lib import *

# Function to initialize the database with separate tables
def sep_initialize_database():
    conn = sqlite3.connect("NDVI_reports_.db")
    cursor = conn.cursor()

    # Create table for Graph Data  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Graph_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            date TEXT,
            ndvi_value REAL,
            report_type TEXT DEFAULT "Graph"
        )
    ''')

    # Create table for Histogram Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Histogram_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            location TEXT,
            data REAL,
            report_type TEXT DEFAULT "Histogram"
        )
    ''')

    conn.commit()
    conn.close()

# Call this function when starting the app to ensure tables exist
sep_initialize_database()
























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
