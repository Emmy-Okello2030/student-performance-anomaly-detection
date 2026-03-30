"""
Database module implementing SDS Section 4.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH

class DatabaseManager:
    """Manages all database operations."""
    
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection."""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        print(f"✅ Connected to database: {self.db_path}")
        return self
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")
    
    def create_tables(self):
        """Create all tables as per SDS Section 4.2."""
        print("\n" + "="*60)
        print("CREATING DATABASE TABLES")
        print("="*60)
        
        # Students table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id_student INTEGER PRIMARY KEY,
                gender TEXT,
                region TEXT,
                highest_education TEXT,
                imd_band TEXT,
                age_band TEXT,
                disability TEXT,
                final_result TEXT,
                is_disabled INTEGER,
                data_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("  ✅ Created table: students")
        
        # Courses table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                code_module TEXT,
                code_presentation TEXT,
                length INTEGER,
                PRIMARY KEY (code_module, code_presentation)
            )
        ''')
        print("  ✅ Created table: courses")
        
        # VLE interactions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vle_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_student INTEGER,
                code_module TEXT,
                code_presentation TEXT,
                date TEXT,
                sum_click INTEGER,
                activity_type TEXT,
                FOREIGN KEY (id_student) REFERENCES students(id_student)
            )
        ''')
        print("  ✅ Created table: vle_interactions")
        
        # Assessments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id_assessment INTEGER PRIMARY KEY,
                code_module TEXT,
                code_presentation TEXT,
                assessment_type TEXT,
                date INTEGER,
                weight REAL,
                FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation)
            )
        ''')
        print("  ✅ Created table: assessments")
        
        # Student assessment table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_assessment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_student INTEGER,
                id_assessment INTEGER,
                score REAL,
                is_banked INTEGER,
                grade_category TEXT,
                FOREIGN KEY (id_student) REFERENCES students(id_student),
                FOREIGN KEY (id_assessment) REFERENCES assessments(id_assessment)
            )
        ''')
        print("  ✅ Created table: student_assessment")
        
        # Risk scores table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_scores (
                student_id INTEGER PRIMARY KEY,
                pred_score REAL,
                anom_score REAL,
                composite_risk REAL,
                risk_level TEXT,
                confidence REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id_student)
            )
        ''')
        print("  ✅ Created table: risk_scores")
        
        # Anomaly flags table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_flags (
                flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                algorithm TEXT,
                anomaly_score REAL,
                threshold REAL,
                flagged_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_contributions TEXT,
                FOREIGN KEY (student_id) REFERENCES students(id_student)
            )
        ''')
        print("  ✅ Created table: anomaly_flags")
        
        # Model metadata table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_type TEXT,
                parameters TEXT,
                accuracy REAL,
                trained_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT
            )
        ''')
        print("  ✅ Created table: model_metadata")
        
        self.connection.commit()
        print("\n✅ All tables created successfully!")
    
    def insert_students(self, df):
        """Insert student data from DataFrame."""
        print(f"\nInserting {len(df)} students...")
        count = 0
        for _, row in df.iterrows():
            try:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO students 
                    (id_student, gender, region, highest_education, imd_band, age_band, 
                     disability, final_result, is_disabled, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('id_student'),
                    row.get('gender'),
                    row.get('region'),
                    row.get('highest_education'),
                    row.get('imd_band'),
                    row.get('age_band'),
                    row.get('disability'),
                    row.get('final_result'),
                    row.get('is_disabled', 0),
                    row.get('data_source', 'oulad')
                ))
                count += 1
            except Exception as e:
                print(f"    Error: {e}")
        
        self.connection.commit()
        print(f"  ✅ Inserted {count} students")
        return count
    
    def insert_risk_score(self, student_id, pred_score, anom_score, composite, risk_level, confidence=0.0):
        """Insert or update risk score."""
        self.cursor.execute('''
            INSERT OR REPLACE INTO risk_scores 
            (student_id, pred_score, anom_score, composite_risk, risk_level, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (student_id, pred_score, anom_score, composite, risk_level, confidence))
        self.connection.commit()
    
    def insert_anomaly_flag(self, student_id, algorithm, score, threshold, features=None):
        """Insert anomaly flag."""
        features_json = json.dumps(features) if features else None
        self.cursor.execute('''
            INSERT INTO anomaly_flags 
            (student_id, algorithm, anomaly_score, threshold, feature_contributions)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, algorithm, score, threshold, features_json))
        self.connection.commit()
    
    def get_high_risk_students(self, threshold=0.7):
        """Get all students with composite risk above threshold."""
        self.cursor.execute('''
            SELECT s.*, r.composite_risk, r.risk_level 
            FROM students s
            JOIN risk_scores r ON s.id_student = r.student_id
            WHERE r.composite_risk > ?
            ORDER BY r.composite_risk DESC
        ''', (threshold,))
        return self.cursor.fetchall()
    
    def get_recent_alerts(self, limit=10):
        """Get most recent anomaly alerts."""
        self.cursor.execute('''
            SELECT af.*, s.gender, s.region 
            FROM anomaly_flags af
            JOIN students s ON af.student_id = s.id_student
            ORDER BY af.flagged_date DESC
            LIMIT ?
        ''', (limit,))
        return self.cursor.fetchall()
    
    def get_table_info(self):
        """Get information about all tables."""
        self.cursor.execute('''
            SELECT name FROM sqlite_master WHERE type='table'
        ''')
        tables = self.cursor.fetchall()
        
        info = {}
        for table in tables:
            table_name = table[0]
            self.cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = self.cursor.fetchone()[0]
            info[table_name] = count
        
        return info
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()