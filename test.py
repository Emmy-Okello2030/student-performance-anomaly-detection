import pandas as pd
from pathlib import Path

# Check what files you have
data_dir = Path('data/processed')
files = list(data_dir.glob('*.csv'))
print("CSV files found:", files)

# Load and check columns
for file in files:
    df = pd.read_csv(file)
    print(f"\n{file.name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
