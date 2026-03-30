"""
Complete File Checker - Shows all issues with your project files
No pkg_resources dependency - uses modern importlib.metadata
"""

import os
import sys
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError

def print_header(text):
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def check_file_exists(path, description):
    if path.exists():
        size = path.stat().st_size
        if size == 0:
            print(f"  ⚠️  {description}: {path} (EMPTY FILE - {size} bytes)")
            return False
        else:
            print(f"  ✅ {description}: {path} ({size:,} bytes)")
            return True
    else:
        print(f"  ❌ {description}: {path} (MISSING!)")
        return False

def check_python_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  Could not check {file_path}: {e}")
        return False

def get_package_version(package_name):
    """Get package version using importlib.metadata."""
    try:
        dist = distribution(package_name)
        return dist.version
    except PackageNotFoundError:
        return None
    except Exception:
        return None

def check_package_installed(package_name, import_name=None):
    """Check if a package is installed by trying to import it."""
    if import_name is None:
        import_name = package_name
    
    # First try importlib.metadata
    version = get_package_version(package_name)
    if version:
        return True, version
    
    # Fallback: try importing the package
    try:
        if package_name == 'scikit-learn':
            import sklearn
            return True, sklearn.__version__ if hasattr(sklearn, '__version__') else "unknown"
        else:
            module = __import__(import_name)
            if hasattr(module, '__version__'):
                return True, module.__version__
            else:
                return True, "unknown"
    except ImportError:
        return False, None

print_header("STUDENT PERFORMANCE ANOMALY DETECTION SYSTEM - FILE CHECKER")

# =========================================================
# Check 1: Required folders
# =========================================================
print_header("CHECKING FOLDER STRUCTURE")

folders = [
    'data/raw/oulad',
    'data/raw/uci',
    'data/processed',
    'models/predictive',
    'models/anomaly',
    'src/data_pipeline',
    'src/models/predictive',
    'src/models/anomaly',
    'src/risk_engine',
    'src/dashboard',
    'src/export',
    'src/utils',
    'scripts',
    'outputs/figures',
    'outputs/reports',
    'outputs/logs'
]

for folder in folders:
    path = Path(folder)
    if path.exists():
        print(f"  ✅ {folder}")
    else:
        print(f"  ❌ {folder} (MISSING!)")

# =========================================================
# Check 2: OULAD dataset files
# =========================================================
print_header("CHECKING OULAD DATASET FILES")

oulad_files = [
    'studentInfo.csv',
    'studentRegistration.csv',
    'studentVle.csv',
    'vle.csv',
    'assessments.csv',
    'studentAssessment.csv'
]

oulad_path = Path('data/raw/oulad')
if oulad_path.exists():
    for file in oulad_files:
        check_file_exists(oulad_path / file, f"OULAD file")
else:
    print(f"  ❌ OULAD folder not found!")

# =========================================================
# Check 3: UCI dataset files
# =========================================================
print_header("CHECKING UCI DATASET FILES")

uci_files = [
    'student-mat.csv',
    'student-por.csv'
]

uci_path = Path('data/raw/uci')
if uci_path.exists():
    for file in uci_files:
        check_file_exists(uci_path / file, f"UCI file")
else:
    print(f"  ❌ UCI folder not found!")

# =========================================================
# Check 4: Python script files
# =========================================================
print_header("CHECKING PYTHON SCRIPTS")

python_files = [
    'src/data_pipeline/complete_cleaner.py',
    'src/data_pipeline/run_pipeline.py',
    'src/models/predictive/random_forest.py',
    'src/models/anomaly/isolation_forest.py',
    'src/risk_engine/integrator.py',
    'src/dashboard/app.py',
    'src/utils/database.py',
    'scripts/init_system.py',
    'run.py',
    'config.py',
    'requirements.txt'
]

all_python_files_exist = True
for file in python_files:
    path = Path(file)
    if path.exists():
        print(f"  ✅ {file}")
        # Check syntax for .py files
        if file.endswith('.py'):
            check_python_syntax(file)
    else:
        print(f"  ❌ {file} (MISSING!)")
        all_python_files_exist = False

# =========================================================
# Check 5: Database file
# =========================================================
print_header("CHECKING DATABASE")

db_path = Path('data/processed/anomaly_detection.db')
if db_path.exists():
    size = db_path.stat().st_size
    print(f"  ✅ Database found: {db_path} ({size:,} bytes)")
    
    # Try to connect to database
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  📊 Tables in database: {[t[0] for t in tables]}")
        conn.close()
    except Exception as e:
        print(f"  ⚠️  Could not read database: {e}")
else:
    print(f"  ❌ Database not found! Run: python scripts/init_system.py")

# =========================================================
# Check 6: Package installations - USING IMPORTLIB (no pkg_resources)
# =========================================================
print_header("CHECKING INSTALLED PACKAGES")

packages_to_check = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('scikit-learn', 'sklearn'),
    ('streamlit', 'streamlit'),
    ('plotly', 'plotly'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('joblib', 'joblib')
]

all_packages_installed = True
for display_name, import_name in packages_to_check:
    installed, version = check_package_installed(display_name, import_name)
    if installed:
        print(f"  ✅ {display_name} {version}")
    else:
        print(f"  ❌ {display_name} (NOT INSTALLED!)")
        all_packages_installed = False

# =========================================================
# Summary
# =========================================================
print_header("SUMMARY")

# Check datasets
datasets_found = True
if not oulad_path.exists():
    print("❌ OULAD folder missing")
    datasets_found = False
if not uci_path.exists():
    print("❌ UCI folder missing")
    datasets_found = False

if datasets_found:
    print("✅ Datasets: FOUND")
else:
    print("❌ Datasets: MISSING - Place your CSV files in data/raw/oulad/ and data/raw/uci/")

# Check Python scripts
if all_python_files_exist:
    print("✅ Python scripts: ALL FOUND")
else:
    print("❌ Python scripts: SOME MISSING - Run the file creation commands")

# Check database
if db_path.exists():
    print("✅ Database: FOUND")
else:
    print("❌ Database: MISSING - Run: python scripts/init_system.py")

# Check packages
if all_packages_installed:
    print("✅ Python packages: ALL INSTALLED")
else:
    print("❌ Python packages: SOME MISSING - Run: pip install -r requirements.txt")

print_header("NEXT STEPS")

if not all_packages_installed:
    print("📦 Install missing packages:")
    print("   pip install plotly")
    if not all_packages_installed:
        print("   pip install -r requirements.txt")
    print()

if not all_python_files_exist:
    print("📝 Create missing Python files using the file creation commands")
    print()

if not db_path.exists():
    print("🗄️  Initialize database:")
    print("   python scripts/init_system.py")
    print()

print("🚀 Once everything is ✅, run your pipeline:")
print("   python src/data_pipeline/run_pipeline.py")
print("   streamlit run src/dashboard/app.py")