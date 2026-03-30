"""
COMPLETE AUTOMATED DATA CLEANING SCRIPT
Student Performance Anomaly Detection System

This script handles ALL cleaning operations for both OULAD and UCI datasets.
No manual intervention required.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CompleteDataCleaner:
    """
    Automated data cleaning for educational datasets.
    Handles everything: missing values, outliers, encoding, merging.
    """
    
    def __init__(self, base_path='data/raw'):
        self.base_path = Path(base_path)
        self.output_path = Path('data/processed')
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Track cleaning operations
        self.cleaning_log = []
        self.issues_found = []
        self.fixes_applied = []
        
    def log_operation(self, message):
        """Log cleaning operations."""
        # Clean message for display
        clean_msg = message.replace('→', '->').replace('✓', '[OK]').replace('⚠', '[WARN]').replace('🔧', '[FIX]')
        print(f"  [OK] {clean_msg}")
        self.cleaning_log.append(message)  # Keep original for file
    
    def log_issue(self, issue):
        """Log issues found."""
        clean_issue = issue.replace('→', '->').replace('✓', '[OK]').replace('⚠', '[WARN]').replace('🔧', '[FIX]')
        print(f"  [WARN] Issue: {clean_issue}")
        self.issues_found.append(issue)  # Keep original for file
    
    def log_fix(self, fix):
        """Log fixes applied."""
        clean_fix = fix.replace('→', '->').replace('✓', '[OK]').replace('⚠', '[WARN]').replace('🔧', '[FIX]')
        print(f"  [FIX] Fix: {clean_fix}")
        self.fixes_applied.append(fix)  # Keep original for file
    
    def clean_oulad(self):
        """
        Complete cleaning pipeline for OULAD dataset.
        Handles all 6 CSV files.
        """
        print("\n" + "="*80)
        print("CLEANING OULAD DATASET")
        print("="*80)
        
        oulad_path = self.base_path / 'oulad'
        if not oulad_path.exists():
            self.log_issue(f"OULAD path not found: {oulad_path}")
            return None
        
        # Dictionary to store cleaned dataframes
        cleaned = {}
        
        # =========================================================
        # 1. CLEAN studentInfo.csv
        # =========================================================
        print("\n--- Cleaning studentInfo.csv ---")
        info_path = oulad_path / 'studentInfo.csv'
        if info_path.exists():
            df_info = pd.read_csv(info_path)
            self.log_operation(f"Loaded studentInfo: ({df_info.shape[0]}, {df_info.shape[1]})")
            
            # Check missing values
            missing = df_info.isnull().sum()
            if missing.sum() > 0:
                missing_dict = {k: v for k, v in missing[missing>0].to_dict().items()}
                self.log_issue(f"Missing values found: {missing_dict}")
                
                # Handle missing values
                # final_result might have missing - drop these rows
                before = len(df_info)
                df_info = df_info.dropna(subset=['final_result'])
                self.log_fix(f"Dropped {before - len(df_info)} rows with missing final_result")
                
                # Fill other missing with appropriate values
                for col in df_info.columns:
                    if df_info[col].dtype == 'object':
                        df_info[col] = df_info[col].fillna('Unknown')
                    else:
                        df_info[col] = df_info[col].fillna(df_info[col].median())
            
            # Check for duplicates
            duplicates = df_info.duplicated().sum()
            if duplicates > 0:
                self.log_issue(f"Found {duplicates} duplicate rows")
                df_info = df_info.drop_duplicates()
                self.log_fix(f"Removed {duplicates} duplicates")
            
            # Standardize categorical values
            categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
            for col in categorical_cols:
                if col in df_info.columns:
                    # Convert to string and strip whitespace
                    df_info[col] = df_info[col].astype(str).str.strip()
                    # Standardize case
                    df_info[col] = df_info[col].str.lower()
                    # Replace common variations
                    df_info[col] = df_info[col].replace({
                        'y': 'yes', 'n': 'no',
                        'true': 'yes', 'false': 'no',
                        '1': 'yes', '0': 'no'
                    })
            
            # Create derived columns
            df_info['is_disabled'] = (df_info['disability'] == 'yes').astype(int)
            
            cleaned['studentInfo'] = df_info
            self.log_operation(f"studentInfo cleaned: ({df_info.shape[0]}, {df_info.shape[1]})")
        else:
            self.log_issue(f"File not found: {info_path}")
        
        # =========================================================
        # 2. CLEAN studentRegistration.csv
        # =========================================================
        print("\n--- Cleaning studentRegistration.csv ---")
        reg_path = oulad_path / 'studentRegistration.csv'
        if reg_path.exists():
            df_reg = pd.read_csv(reg_path)
            self.log_operation(f"Loaded studentRegistration: ({df_reg.shape[0]}, {df_reg.shape[1]})")
            
            # Handle date columns
            date_cols = ['date_registration', 'date_unregistration']
            for col in date_cols:
                if col in df_reg.columns:
                    # Convert to datetime, coerce errors to NaT
                    df_reg[col] = pd.to_datetime(df_reg[col], errors='coerce')
                    
                    # Calculate registration duration
                    if col == 'date_registration':
                        df_reg['registration_day'] = df_reg[col].dt.dayofyear
                    elif col == 'date_unregistration':
                        # Create flag for unregistered
                        df_reg['is_unregistered'] = df_reg[col].notna().astype(int)
            
            # Calculate days enrolled
            if 'date_registration' in df_reg.columns:
                # Use a reference date (start of course)
                df_reg['days_enrolled'] = df_reg.apply(
                    lambda row: (pd.Timestamp('2014-01-01') - row['date_registration']).days 
                    if pd.notna(row['date_registration']) else 0,
                    axis=1
                )
                # Clip negative values
                df_reg['days_enrolled'] = df_reg['days_enrolled'].clip(lower=0)
            
            cleaned['studentRegistration'] = df_reg
            self.log_operation(f"studentRegistration cleaned: ({df_reg.shape[0]}, {df_reg.shape[1]})")
        
        # =========================================================
        # 3. CLEAN studentVle.csv (Largest file - 10M+ rows)
        # =========================================================
        print("\n--- Cleaning studentVle.csv (this may take a while) ---")
        vle_path = oulad_path / 'studentVle.csv'
        if vle_path.exists():
            # Use chunks for large file
            chunks = []
            for chunk in pd.read_csv(vle_path, chunksize=100000):
                # Clean each chunk
                
                # Remove impossible click counts
                chunk = chunk[chunk['sum_click'] >= 0]
                chunk = chunk[chunk['sum_click'] < 10000]  # Remove extreme outliers
                
                # Create engagement features
                chunk['log_clicks'] = np.log1p(chunk['sum_click'])
                chunk['is_high_engagement'] = (chunk['sum_click'] > chunk['sum_click'].quantile(0.9)).astype(int)
                chunk['is_low_engagement'] = (chunk['sum_click'] < chunk['sum_click'].quantile(0.1)).astype(int)
                
                chunks.append(chunk)
            
            if chunks:
                df_vle = pd.concat(chunks, ignore_index=True)
                
                # Aggregate per student for features
                student_vle_stats = df_vle.groupby('id_student').agg({
                    'sum_click': ['mean', 'std', 'sum', 'count'],
                    'is_high_engagement': 'sum',
                    'is_low_engagement': 'sum'
                }).round(2)
                
                # Flatten column names
                student_vle_stats.columns = ['_'.join(col).strip() for col in student_vle_stats.columns.values]
                student_vle_stats = student_vle_stats.reset_index()
                
                cleaned['studentVle'] = student_vle_stats
                self.log_operation(f"studentVle cleaned: {len(df_vle):,} rows -> {len(student_vle_stats)} student summaries")
            else:
                self.log_issue("No valid data in studentVle")
        else:
            self.log_issue(f"File not found: {vle_path}")
        
        # =========================================================
        # 4. CLEAN vle.csv (course materials)
        # =========================================================
        print("\n--- Cleaning vle.csv ---")
        vle_materials_path = oulad_path / 'vle.csv'
        if vle_materials_path.exists():
            df_materials = pd.read_csv(vle_materials_path)
            
            # Clean activity types
            df_materials['activity_type'] = df_materials['activity_type'].str.lower().str.strip()
            
            # One-hot encode activity types
            activity_dummies = pd.get_dummies(df_materials['activity_type'], prefix='activity')
            df_materials = pd.concat([df_materials, activity_dummies], axis=1)
            
            cleaned['vle'] = df_materials
            self.log_operation(f"vle cleaned: ({df_materials.shape[0]}, {df_materials.shape[1]})")
        
        # =========================================================
        # 5. CLEAN assessments.csv
        # =========================================================
        print("\n--- Cleaning assessments.csv ---")
        ass_path = oulad_path / 'assessments.csv'
        if ass_path.exists():
            df_ass = pd.read_csv(ass_path)
            
            # Clean assessment types
            df_ass['assessment_type'] = df_ass['assessment_type'].str.lower().str.strip()
            
            # Handle dates
            df_ass['date'] = pd.to_datetime(df_ass['date'], errors='coerce')
            df_ass['week'] = df_ass['date'].dt.isocalendar().week
            
            # One-hot encode assessment types
            ass_dummies = pd.get_dummies(df_ass['assessment_type'], prefix='assessment')
            df_ass = pd.concat([df_ass, ass_dummies], axis=1)
            
            cleaned['assessments'] = df_ass
            self.log_operation(f"assessments cleaned: ({df_ass.shape[0]}, {df_ass.shape[1]})")
        
        # =========================================================
        # 6. CLEAN studentAssessment.csv
        # =========================================================
        print("\n--- Cleaning studentAssessment.csv ---")
        student_ass_path = oulad_path / 'studentAssessment.csv'
        if student_ass_path.exists():
            df_student_ass = pd.read_csv(student_ass_path)
            
            # Handle scores
            # Score is between 0-100, but some might be missing
            df_student_ass = df_student_ass.dropna(subset=['score'])
            df_student_ass = df_student_ass[(df_student_ass['score'] >= 0) & (df_student_ass['score'] <= 100)]
            
            # Create grade categories
            df_student_ass['grade_category'] = pd.cut(
                df_student_ass['score'],
                bins=[0, 40, 60, 75, 100],
                labels=['Failing', 'Passing', 'Good', 'Excellent']
            )
            
            cleaned['studentAssessment'] = df_student_ass
            self.log_operation(f"studentAssessment cleaned: ({df_student_ass.shape[0]}, {df_student_ass.shape[1]})")
        
        # =========================================================
        # MERGE ALL OULAD TABLES
        # =========================================================
        print("\n--- Merging OULAD tables ---")
        
        merged = None
        
        # Start with studentInfo
        if 'studentInfo' in cleaned:
            merged = cleaned['studentInfo'].copy()
            
            # Merge with registration data
            if 'studentRegistration' in cleaned:
                merged = pd.merge(
                    merged, 
                    cleaned['studentRegistration'],
                    on=['id_student', 'code_module', 'code_presentation'],
                    how='left'
                )
            
            # Merge with VLE statistics
            if 'studentVle' in cleaned:
                merged = pd.merge(
                    merged,
                    cleaned['studentVle'],
                    on='id_student',
                    how='left'
                )
            
            # Merge with assessment data (aggregated)
            if 'studentAssessment' in cleaned:
                # Aggregate per student
                student_scores = cleaned['studentAssessment'].groupby('id_student').agg({
                    'score': ['mean', 'std', 'min', 'max', 'count']
                }).round(2)
                student_scores.columns = ['_'.join(col).strip() for col in student_scores.columns.values]
                student_scores = student_scores.reset_index()
                
                merged = pd.merge(
                    merged,
                    student_scores,
                    on='id_student',
                    how='left'
                )
            
            # Fill any missing values from merges
            numeric_cols = merged.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if merged[col].isnull().sum() > 0:
                    merged[col] = merged[col].fillna(merged[col].median())
            
            cleaned['merged_oulad'] = merged
            self.log_operation(f"OULAD merged dataset: ({merged.shape[0]}, {merged.shape[1]})")
        
        return cleaned
    
    def clean_uci(self):
        """
        Complete cleaning pipeline for UCI dataset.
        Handles both Math and Portuguese datasets.
        """
        print("\n" + "="*80)
        print("CLEANING UCI DATASET")
        print("="*80)
        
        uci_path = self.base_path / 'uci'
        if not uci_path.exists():
            self.log_issue(f"UCI path not found: {uci_path}")
            return None
        
        cleaned = {}
        
        # =========================================================
        # 1. CLEAN student-mat.csv
        # =========================================================
        print("\n--- Cleaning student-mat.csv ---")
        math_path = uci_path / 'student-mat.csv'
        if math_path.exists():
            df_math = pd.read_csv(math_path, sep=';')
            df_math['subject'] = 'math'
            df_math = self._clean_uci_dataframe(df_math, 'math')
            cleaned['math'] = df_math
            self.log_operation(f"Math dataset cleaned: ({df_math.shape[0]}, {df_math.shape[1]})")
        
        # =========================================================
        # 2. CLEAN student-por.csv
        # =========================================================
        print("\n--- Cleaning student-por.csv ---")
        por_path = uci_path / 'student-por.csv'
        if por_path.exists():
            df_por = pd.read_csv(por_path, sep=';')
            df_por['subject'] = 'portuguese'
            df_por = self._clean_uci_dataframe(df_por, 'portuguese')
            cleaned['portuguese'] = df_por
            self.log_operation(f"Portuguese dataset cleaned: ({df_por.shape[0]}, {df_por.shape[1]})")
        
        # =========================================================
        # 3. MERGE UCI DATASETS
        # =========================================================
        print("\n--- Merging UCI datasets ---")
        
        if 'math' in cleaned and 'portuguese' in cleaned:
            # Concatenate both subjects
            merged_uci = pd.concat([cleaned['math'], cleaned['portuguese']], ignore_index=True)
            cleaned['merged_uci'] = merged_uci
            self.log_operation(f"UCI merged dataset: ({merged_uci.shape[0]}, {merged_uci.shape[1]})")
        
        return cleaned
    
    def _clean_uci_dataframe(self, df, subject_name):
        """
        Clean individual UCI dataframe.
        """
        original_shape = df.shape
        
        # =========================================================
        # Handle missing values
        # =========================================================
        # Check for 'NA' strings and replace with NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('NA', np.nan)
        
        # Report missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_dict = {k: v for k, v in missing[missing>0].to_dict().items()}
            self.log_issue(f"{subject_name}: Missing values found: {missing_dict}")
            
            # For categorical columns, fill with mode
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    self.log_fix(f"{subject_name}: Filled {col} with mode: {mode_val}")
            
            # For numeric columns, fill with median
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.log_fix(f"{subject_name}: Filled {col} with median: {median_val:.2f}")
        
        # =========================================================
        # Encode categorical variables
        # =========================================================
        categorical_mappings = {
            'school': {'GP': 0, 'MS': 1},
            'sex': {'F': 0, 'M': 1},
            'address': {'U': 0, 'R': 1},  # Urban=0, Rural=1
            'famsize': {'LE3': 0, 'GT3': 1},  # ≤3=0, >3=1
            'Pstatus': {'T': 0, 'A': 1},  # Together=0, Apart=1
            'schoolsup': {'no': 0, 'yes': 1},
            'famsup': {'no': 0, 'yes': 1},
            'paid': {'no': 0, 'yes': 1},
            'activities': {'no': 0, 'yes': 1},
            'nursery': {'no': 0, 'yes': 1},
            'higher': {'no': 0, 'yes': 1},
            'internet': {'no': 0, 'yes': 1},
            'romantic': {'no': 0, 'yes': 1}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                self.log_fix(f"{subject_name}: Encoded {col}")
        
        # =========================================================
        # Handle ordered categoricals
        # =========================================================
        # Mother's education (Medu) and Father's education (Fedu)
        # 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 
        # 3 - secondary education, 4 - higher education
        edu_cols = ['Medu', 'Fedu']
        for col in edu_cols:
            if col in df.columns:
                # Ensure within range
                df[col] = df[col].clip(0, 4)
                self.log_fix(f"{subject_name}: Validated {col} range")
        
        # =========================================================
        # Create derived features
        # =========================================================
        # Average of G1, G2, G3
        if all(col in df.columns for col in ['G1', 'G2', 'G3']):
            df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1).round(2)
            df['G_trend'] = (df['G3'] - df['G1']) / 20  # Normalized trend
            
            # Final pass/fail (G3 >= 10)
            df['passed'] = (df['G3'] >= 10).astype(int)
            
            self.log_fix(f"{subject_name}: Created G_avg, G_trend, passed")
        
        # Total alcohol consumption
        if all(col in df.columns for col in ['Dalc', 'Walc']):
            df['total_alcohol'] = df['Dalc'] + df['Walc']
            self.log_fix(f"{subject_name}: Created total_alcohol")
        
        # Family relationship score
        if all(col in df.columns for col in ['famrel', 'freetime', 'goout']):
            df['social_score'] = (df['famrel'] + df['freetime'] + df['goout']) / 3
            self.log_fix(f"{subject_name}: Created social_score")
        
        # Health issues
        if 'health' in df.columns:
            df['has_health_issues'] = (df['health'] < 3).astype(int)
            self.log_fix(f"{subject_name}: Created has_health_issues")
        
        # =========================================================
        # Remove outliers
        # =========================================================
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR  # More lenient: 3*IQR
            upper = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                # Cap instead of remove
                df[col] = df[col].clip(lower, upper)
                self.log_fix(f"{subject_name}: Capped {outliers} outliers in {col}")
        
        # =========================================================
        # Final quality checks
        # =========================================================
        rows_removed = original_shape[0] - df.shape[0]
        if rows_removed > 0:
            self.log_fix(f"{subject_name}: Removed {rows_removed} rows")
        
        return df
    
    def create_master_dataset(self, oulad_data, uci_data):
        """
        Create a unified master dataset from both sources.
        Standardizes column names and formats.
        """
        print("\n" + "="*80)
        print("CREATING MASTER DATASET")
        print("="*80)
        
        master_data = {}
        
        # =========================================================
        # Process OULAD data
        # =========================================================
        if oulad_data and 'merged_oulad' in oulad_data:
            df_oulad = oulad_data['merged_oulad'].copy()
            
            # Standardize column names
            df_oulad.columns = [col.lower().replace(' ', '_') for col in df_oulad.columns]
            
            # Add source identifier
            df_oulad['data_source'] = 'oulad'
            
            # Rename common columns for consistency
            rename_map = {
                'id_student': 'student_id',
                'gender': 'gender_encoded',
                'final_result': 'target_grade'  # This is categorical in OULAD
            }
            df_oulad = df_oulad.rename(columns=rename_map)
            
            master_data['oulad'] = df_oulad
            self.log_operation(f"Processed OULAD for master: ({df_oulad.shape[0]}, {df_oulad.shape[1]})")
        
        # =========================================================
        # Process UCI data
        # =========================================================
        if uci_data and 'merged_uci' in uci_data:
            df_uci = uci_data['merged_uci'].copy()
            
            # Standardize column names
            df_uci.columns = [col.lower() for col in df_uci.columns]
            
            # Add source identifier
            df_uci['data_source'] = 'uci'
            
            # Rename common columns
            rename_map = {
                'g3': 'target_grade',  # Final grade
                'passed': 'target_passed',
                'sex': 'gender_encoded'
            }
            df_uci = df_uci.rename(columns=rename_map)
            
            master_data['uci'] = df_uci
            self.log_operation(f"Processed UCI for master: ({df_uci.shape[0]}, {df_uci.shape[1]})")
        
        # =========================================================
        # Create combined dataset (if needed)
        # =========================================================
        if 'oulad' in master_data and 'uci' in master_data:
            # This is tricky because they have different schemas
            # For now, keep separate
            self.log_operation("Datasets kept separate due to different schemas")
        
        return master_data
    
    def generate_cleaning_report(self):
        """
        Generate a comprehensive cleaning report.
        """
        print("\n" + "="*80)
        print("CLEANING SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📊 Operations performed: {len(self.cleaning_log)}")
        for i, op in enumerate(self.cleaning_log, 1):
            # Clean for display
            clean_op = op.replace('→', '->').replace('✓', '').replace('⚠', '').replace('🔧', '')
            print(f"  {i}. {clean_op}")
        
        print(f"\n⚠️ Issues found: {len(self.issues_found)}")
        for i, issue in enumerate(self.issues_found, 1):
            clean_issue = issue.replace('→', '->').replace('✓', '').replace('⚠', '').replace('🔧', '')
            print(f"  {i}. {clean_issue}")
        
        print(f"\n🔧 Fixes applied: {len(self.fixes_applied)}")
        for i, fix in enumerate(self.fixes_applied, 1):
            clean_fix = fix.replace('→', '->').replace('✓', '').replace('⚠', '').replace('🔧', '')
            print(f"  {i}. {clean_fix}")
        
        # Save report to file with UTF-8 encoding
        report_path = self.output_path / 'cleaning_report.txt'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DATA CLEANING REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"OPERATIONS PERFORMED ({len(self.cleaning_log)}):\n")
                for op in self.cleaning_log:
                    f.write(f"  • {op}\n")
                
                f.write(f"\nISSUES FOUND ({len(self.issues_found)}):\n")
                for issue in self.issues_found:
                    f.write(f"  • {issue}\n")
                
                f.write(f"\nFIXES APPLIED ({len(self.fixes_applied)}):\n")
                for fix in self.fixes_applied:
                    f.write(f"  • {fix}\n")
            
            print(f"\n📄 Report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Error saving report: {e}")
            # Fallback: save without Unicode
            with open(report_path, 'w', encoding='ascii', errors='ignore') as f:
                f.write("="*80 + "\n")
                f.write("DATA CLEANING REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"OPERATIONS PERFORMED ({len(self.cleaning_log)}):\n")
                for op in self.cleaning_log:
                    clean_op = op.encode('ascii', 'ignore').decode('ascii')
                    f.write(f"  • {clean_op}\n")
                
                f.write(f"\nISSUES FOUND ({len(self.issues_found)}):\n")
                for issue in self.issues_found:
                    clean_issue = issue.encode('ascii', 'ignore').decode('ascii')
                    f.write(f"  • {clean_issue}\n")
                
                f.write(f"\nFIXES APPLIED ({len(self.fixes_applied)}):\n")
                for fix in self.fixes_applied:
                    clean_fix = fix.encode('ascii', 'ignore').decode('ascii')
                    f.write(f"  • {clean_fix}\n")
    
    def save_cleaned_data(self, master_data):
        """
        Save all cleaned datasets to CSV.
        """
        print("\n" + "="*80)
        print("SAVING CLEANED DATA")
        print("="*80)
        
        for name, df in master_data.items():
            output_file = self.output_path / f"{name}_cleaned.csv"
            df.to_csv(output_file, index=False)
            print(f"  [OK] Saved {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"       File: {output_file}")
            
            # Also save sample for quick inspection
            sample_file = self.output_path / f"{name}_sample.csv"
            df.head(100).to_csv(sample_file, index=False)
            print(f"  [OK] Saved sample (100 rows)")
            print(f"       File: {sample_file}")
    
    def run_complete_cleaning(self):
        """
        Run the complete cleaning pipeline.
        """
        print("\n" + "="*80)
        print("COMPLETE DATA CLEANING PIPELINE")
        print("Student Performance Anomaly Detection System")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Clean OULAD
        oulad_cleaned = self.clean_oulad()
        
        # Step 2: Clean UCI
        uci_cleaned = self.clean_uci()
        
        # Step 3: Create master dataset
        master_data = self.create_master_dataset(oulad_cleaned, uci_cleaned)
        
        # Step 4: Save everything
        self.save_cleaned_data(master_data)
        
        # Step 5: Generate report
        self.generate_cleaning_report()
        
        print("\n" + "="*80)
        print(f"CLEANING COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return master_data


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Create cleaner instance
    cleaner = CompleteDataCleaner(base_path='data/raw')
    
    # Run complete cleaning
    cleaned_data = cleaner.run_complete_cleaning()
    
    print("\n✅ Data cleaning complete!")
    print("   Check 'data/processed/' for cleaned files")
    print("   Check 'data/processed/cleaning_report.txt' for details")
    