"""
Complete training script - trains all models and saves to database.
Handles both predictive modeling and anomaly detection.
Fixed to work with actual UCI and OULAD data structures.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_SEED
from src.models.predictive.random_forest import RandomForestModel
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.risk_engine.integrator import RiskIntegrator, RiskExplainer
from src.utils.database import DatabaseManager


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)


def load_cleaned_data():
    """Load cleaned datasets."""
    data = {}
    
    uci_path = PROCESSED_DATA_DIR / 'uci_cleaned.csv'
    if uci_path.exists():
        df = pd.read_csv(uci_path)
        data['uci'] = df
        print(f"✅ Loaded UCI data: {df.shape}")
    else:
        print(f"⚠️  UCI data not found at {uci_path}")
    
    oulad_path = PROCESSED_DATA_DIR / 'oulad_cleaned.csv'
    if oulad_path.exists():
        df = pd.read_csv(oulad_path)
        data['oulad'] = df
        print(f"✅ Loaded OULAD data: {df.shape}")
    else:
        print(f"⚠️  OULAD data not found at {oulad_path}")
    
    return data


def prepare_uci_data(df):
    """
    Prepare UCI dataset for training.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (X, y, feature_cols, student_ids)
    """
    print("\n--- Preparing UCI Data ---")
    
    # Get student IDs (use index if no id column)
    student_ids = np.arange(len(df))
    
    # Determine target column
    # UCI dataset has 'target_passed' column (0 or 1)
    if 'target_passed' in df.columns:
        y = df['target_passed'].copy()
        print(f"Using 'target_passed' as target column")
    elif 'failed' in df.columns:
        y = (~df['failed']).astype(int)  # Convert failed to passed
        print(f"Using 'failed' column (inverted) as target")
    elif 'target_grade' in df.columns:
        # Convert grade to pass/fail (assume grade >= 10 is pass)
        y = (df['target_grade'] >= 10).astype(int)
        print(f"Using 'target_grade' >= 10 as target")
    else:
        raise ValueError("No target column found. Expected 'target_passed', 'failed', or 'target_grade'")
    
    # Select features (exclude target and non-numeric columns)
    exclude_cols = {
        'target_passed', 'target_grade', 'failed', 'data_source', 
        'student_id', 'mjob', 'fjob', 'reason', 'guardian', 'subject'
    }
    
    # Get numeric features only
    X = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    X = X[feature_cols]
    
    print(f"Features selected: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:5]}")
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    student_ids = student_ids[mask]
    
    print(f"After removing NaN: X shape {X.shape}, y shape {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols, student_ids


def prepare_oulad_data(df):
    """
    Prepare OULAD dataset for training.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (X, y, feature_cols, student_ids)
    """
    print("\n--- Preparing OULAD Data ---")
    
    # Get student IDs
    student_ids = df['student_id'].values if 'student_id' in df.columns else np.arange(len(df))
    
    # Determine target column
    # OULAD has 'target_grade' with values like 'Pass', 'Fail', 'Withdrawn', 'Distinction'
    if 'target_grade' in df.columns:
        # Convert to binary: Pass/Distinction = 1, Fail/Withdrawn = 0
        y = df['target_grade'].isin(['Pass', 'Distinction']).astype(int)
        print(f"Using 'target_grade' as target (Pass/Distinction=1, else=0)")
    else:
        raise ValueError("OULAD data must have 'target_grade' column")
    
    # Select features (exclude target and non-numeric columns)
    exclude_cols = {
        'target_grade', 'data_source', 'student_id', 'code_module', 
        'code_presentation', 'gender_encoded', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability', 'date_registration', 
        'date_unregistration'
    }
    
    # Get numeric features only
    X = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    X = X[feature_cols]
    
    print(f"Features selected: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:5]}")
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    if isinstance(student_ids, np.ndarray):
        student_ids = student_ids[mask]
    else:
        student_ids = np.arange(len(X))
    
    print(f"After removing NaN: X shape {X.shape}, y shape {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols, student_ids


def train_models_on_dataset(dataset_name, X, y, student_ids, feature_cols):
    """
    Train both predictive and anomaly detection models on a dataset.
    
    Args:
        dataset_name: Name of dataset (uci, oulad)
        X: Features
        y: Target
        student_ids: Student IDs
        feature_cols: Feature column names
    
    Returns:
        Tuple of (rf_model, if_model, X_test, y_test, test_indices)
    """
    print_header(f"TRAINING ON {dataset_name.upper()} DATASET")
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )
    
    print(f"\nTrain: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Train Random Forest (predictive)
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train, X_val, y_val)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    print(f"\nTest Metrics:")
    for metric, value in rf_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    # Show feature importance
    importance = rf_model.get_feature_importance(top_n=5)
    print(f"\nTop 5 Features:")
    print(importance.to_string())
    
    # Save model
    rf_model_path = MODEL_DIR / 'predictive' / f'random_forest_{dataset_name}.pkl'
    rf_model_path.parent.mkdir(parents=True, exist_ok=True)
    rf_model.save_model(str(rf_model_path))
    print(f"Model saved to {rf_model_path}")
    
    # Train Isolation Forest (anomaly detection)
    print("\n--- Training Isolation Forest ---")
    if_model = IsolationForestModel()
    if_model.fit(X)
    if_model_path = MODEL_DIR / 'anomaly' / f'isolation_forest_{dataset_name}.pkl'
    if_model_path.parent.mkdir(parents=True, exist_ok=True)
    if_model.save_model(str(if_model_path))
    print(f"Model saved to {if_model_path}")
    
    print("✅ Models trained successfully!")
    
    return rf_model, if_model, X_test, y_test, X_test.index.values


def save_predictions_to_database(dataset_name, rf_model, if_model, X_test, y_test, 
                                 test_indices, student_ids, rf_metrics):
    """
    Save model predictions and risk scores to database.
    
    Args:
        dataset_name: Name of dataset
        rf_model: Trained Random Forest model
        if_model: Trained Isolation Forest model
        X_test: Test features
        y_test: Test labels
        test_indices: Indices of test samples
        student_ids: Student IDs
        rf_metrics: Random Forest metrics
    """
    print("\n--- Saving to Database ---")
    
    try:
        with DatabaseManager() as db:
            # Save model metadata
            db.cursor.execute('''
                INSERT INTO model_metadata (model_name, model_type, parameters, accuracy, file_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                f'Random Forest ({dataset_name})',
                'classification',
                str(rf_model.params),
                rf_metrics.get('accuracy', 0),
                str(MODEL_DIR / 'predictive' / f'random_forest_{dataset_name}.pkl')
            ))
            
            db.cursor.execute('''
                INSERT INTO model_metadata (model_name, model_type, parameters, accuracy, file_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                f'Isolation Forest ({dataset_name})',
                'anomaly_detection',
                str(if_model.params),
                0.0,
                str(MODEL_DIR / 'anomaly' / f'isolation_forest_{dataset_name}.pkl')
            ))
            
            # Get predictions and anomaly scores
            pred_proba = rf_model.predict_proba(X_test)[:, 1]
            anom_scores = if_model.score_samples(X_test)
            anom_scores_norm = if_model.get_anomaly_scores_normalized(X_test)
            
            # Calculate composite risk scores
            risk_integrator = RiskIntegrator()
            composite_scores = risk_integrator.calculate_composite(pred_proba, anom_scores_norm)
            
            # Save risk scores
            print(f"Saving {len(X_test)} risk scores...")
            saved_count = 0
            
            for i, test_idx in enumerate(test_indices):
                try:
                    # Get student ID
                    if student_ids is not None and i < len(student_ids):
                        student_id = int(student_ids[test_idx]) if test_idx < len(student_ids) else i
                    else:
                        student_id = i
                    
                    pred_score = float(pred_proba[i])
                    anom_score = float(anom_scores_norm[i])
                    composite = float(composite_scores[i])
                    risk_level, _ = risk_integrator.classify_risk(composite)
                    
                    db.insert_risk_score(
                        student_id=student_id,
                        pred_score=pred_score,
                        anom_score=anom_score,
                        composite=composite,
                        risk_level=risk_level,
                        confidence=0.85
                    )
                    saved_count += 1
                    
                    # Save anomaly flags for high anomaly scores
                    if anom_score > 0.7:
                        db.insert_anomaly_flag(
                            student_id=student_id,
                            algorithm='isolation_forest',
                            score=anom_score,
                            threshold=0.7,
                            features={'dataset': dataset_name}
                        )
                
                except Exception as e:
                    print(f"  Error saving score for student {i}: {e}")
            
            print(f"✅ Saved {saved_count} risk scores to database")
    
    except Exception as e:
        print(f"⚠️  Error saving to database: {e}")
        print("Continuing without database save...")


def main():
    """Main training function."""
    print_header("STUDENT PERFORMANCE ANOMALY DETECTION - MODEL TRAINING")
    
    # Load data
    data = load_cleaned_data()
    
    if not data:
        print("\n❌ No cleaned data found!")
        print("Please run the data pipeline first or ensure CSV files exist in data/processed/")
        sys.exit(1)
    
    # Train on each dataset
    for dataset_name, df in data.items():
        try:
            # Prepare data based on dataset type
            if dataset_name == 'uci':
                X, y, feature_cols, student_ids = prepare_uci_data(df)
            elif dataset_name == 'oulad':
                X, y, feature_cols, student_ids = prepare_oulad_data(df)
            else:
                print(f"⚠️  Unknown dataset: {dataset_name}")
                continue
            
            if len(X) == 0:
                print(f"\n⚠️  No valid data for {dataset_name} dataset after preprocessing")
                continue
            
            # Train models
            rf_model, if_model, X_test, y_test, test_indices = train_models_on_dataset(
                dataset_name, X, y, student_ids, feature_cols
            )
            
            # Get metrics for saving
            rf_metrics = rf_model.evaluate(X_test, y_test)
            
            # Save to database
            save_predictions_to_database(
                dataset_name, rf_model, if_model, X_test, y_test, 
                test_indices, student_ids, rf_metrics
            )
        
        except Exception as e:
            print(f"\n❌ Error training on {dataset_name} dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print_header("TRAINING COMPLETE")
    print("\n✅ All models trained and saved successfully!")
    print("\nNext step: Launch dashboard with 'streamlit run src/dashboard/app.py'")


if __name__ == "__main__":
    main()
