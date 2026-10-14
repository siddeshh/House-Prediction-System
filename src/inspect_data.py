import pandas as pd
import numpy as np

def is_float(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def inspect_dataset(file_path: str):
    print("=" * 60)
    print("PUNE HOUSE PRICE PREDICTION - DATASET INSPECTION REPORT")
    print("=" * 60)
    
    # 1. Load dataset
    df = pd.read_csv(file_path)
    
    # 2. Shape
    print(f"\n1. Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # 3. Columns & Data Types
    print("\n2. Columns & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:15s}: {str(dtype):10s} (Non-null: {df[col].notnull().sum():,}/{len(df):,})")
        
    # 4. Missing Values
    print("\n3. Missing Value Counts:")
    missing = df.isnull().sum()
    for col, val in missing[missing > 0].items():
        pct = (val / len(df)) * 100
        print(f"   - {col:15s}: {val:,} missing ({pct:.2f}%)")
        
    # 5. First 5 rows
    print("\n4. Sample Data (First 3 rows):")
    print(df.head(3).to_string())
    
    # 6. Cardinality of Locality
    print(f"\n5. Locality (site_location): {df['site_location'].nunique()} unique localities")
    print("   Top 10 localities by listing count:")
    for loc, count in df['site_location'].value_counts().head(10).items():
        print(f"     * {loc}: {count} listings")
        
    # 7. Unique values in 'size'
    print(f"\n6. Unique Property Sizes ({df['size'].nunique()} variations):")
    print("  ", list(df['size'].dropna().unique()))
    
    # 8. Non-float values in 'total_sqft'
    non_float = df[~df['total_sqft'].apply(is_float)]['total_sqft']
    print(f"\n7. Non-standard total_sqft entries: {len(non_float):,} rows")
    print("   Sample anomalies:")
    for sample in non_float.head(10).tolist():
        print(f"     * '{sample}'")
        
    # 9. Summary stats of price and numeric cols
    print("\n8. Numerical Summary Statistics:")
    print(df.describe().to_string())
    
    return df

if __name__ == "__main__":
    df = inspect_dataset("data/raw/Pune house data.csv")
