"""
Initialize the complete system - creates database and folders.
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("SYSTEM INITIALIZATION")
print("="*80)

# Create necessary folders
folders = [
    'data/raw/oulad',
    'data/raw/uci',
    'data/processed',
    'models/predictive',
    'models/anomaly',
    'outputs/figures',
    'outputs/reports',
    'outputs/logs'
]

print("\n📁 Creating folders...")
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {folder}")

# Create database
print("\n🗄️ Creating database...")
try:
    import sqlite3
    conn = sqlite3.connect('data/processed/anomaly_detection.db')
    
    # Create tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id_student INTEGER PRIMARY KEY,
            gender TEXT,
            region TEXT,
            highest_education TEXT,
            final_result TEXT,
            data_source TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS risk_scores (
            student_id INTEGER PRIMARY KEY,
            pred_score REAL,
            anom_score REAL,
            composite_risk REAL,
            risk_level TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_flags (
            flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            algorithm TEXT,
            anomaly_score REAL,
            flagged_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("  ✅ Database created with tables")
except Exception as e:
    print(f"  ❌ Error creating database: {e}")

print("\n" + "="*80)
print("✅ SYSTEM READY!")
print("="*80)