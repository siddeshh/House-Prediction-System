import re
import numpy as np
import pandas as pd
from typing import Optional, Union

# Conversion factors to Square Feet
UNIT_CONVERSIONS = {
    'sq. meter': 10.7639,
    'sq. meters': 10.7639,
    'sq. yard': 9.0,
    'sq. yards': 9.0,
    'acres': 43560.0,
    'acre': 43560.0,
    'guntha': 1089.0,
    'perch': 272.25,
    'grounds': 2400.0,
    'cents': 435.6
}

def parse_bhk(size_val: Union[str, float, None]) -> Optional[int]:
    """
    Extracts the integer count of bedrooms/BHK from a raw size string.
    Handles '2 BHK', '4 Bedroom', '1 RK', NaN, etc.
    """
    if pd.isna(size_val):
        return None
    val_str = str(size_val).strip()
    match = re.match(r"^(\d+)", val_str)
    if match:
        return int(match.group(1))
    return None

def parse_total_sqft(val: Union[str, float, int, None]) -> Optional[float]:
    """
    Converts diverse total_sqft representations into a standard float (in square feet).
    Handles:
      - Clean numeric values: '1056' -> 1056.0
      - Numerical ranges: '2100 - 2850' -> (2100 + 2850) / 2 = 2475.0
      - Documented unit conversions: '34.46Sq. Meter' -> 34.46 * 10.7639
      - Invalid/unparseable values -> None
    """
    if pd.isna(val):
        return None
    
    val_str = str(val).strip()
    
    # 1. Check for standard float/integer
    try:
        return float(val_str)
    except ValueError:
        pass
        
    # 2. Check for ranges like '2100 - 2850'
    if '-' in val_str:
        tokens = [t.strip() for t in val_str.split('-')]
        if len(tokens) == 2:
            try:
                low = float(tokens[0])
                high = float(tokens[1])
                return (low + high) / 2.0
            except ValueError:
                pass
                
    # 3. Check for unit strings (e.g. '34.46Sq. Meter', '1100Sq. Yards')
    val_lower = val_str.lower()
    for unit, factor in UNIT_CONVERSIONS.items():
        if unit in val_lower:
            # Extract the numeric part preceding or surrounding the unit
            num_match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", val_lower)
            if num_match:
                try:
                    num = float(num_match.group(0))
                    return round(num * factor, 2)
                except ValueError:
                    pass
                    
    return None

def clean_locality(loc: Union[str, float, None]) -> str:
    """
    Standardizes locality names by stripping whitespace and normalizing case.
    """
    if pd.isna(loc):
        return 'other'
    cleaned = str(loc).strip()
    return cleaned if cleaned else 'other'

def remove_bhk_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes properties that violate basic physical reality:
    A property must have at least 300 sq.ft. per bedroom.
    (e.g., 43 bedrooms in 2400 sq.ft = 55 sq.ft/bhk is an obvious data error).
    """
    return df[~(df['total_sqft'] / df['bhk'] < 300)].copy()

def remove_bath_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain rule: It is unusual for a residential property to have more bathrooms
    than bedrooms + 2 (e.g., a 2 BHK with 5 bathrooms).
    """
    return df[df['bath'] <= (df['bhk'] + 2)].copy()

def remove_pps_outliers_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes extreme price-per-square-foot anomalies per locality using 1-standard-deviation bounds.
    Preserves legitimate listings while filtering data entry typos.
    """
    df_out = pd.DataFrame()
    for _, subdf in df.groupby('site_location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        # Keep properties within 1 standard deviation of locality mean
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def clean_housing_data(raw_csv_path: str, processed_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    End-to-end data cleaning pipeline for Pune house data.
    Preserves raw CSV immutability and outputs clean structured data.
    """
    df = pd.read_csv(raw_csv_path)
    
    # 1. Drop features with high missingness or irrelevant cardinality
    # 'society' has 41% missing and opaque hashes; 'availability' has ~80+ date categories
    cols_to_drop = [c for c in ['society', 'availability'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # 2. Parse BHK from size
    df['bhk'] = df['size'].apply(parse_bhk)
    df = df.drop(columns=['size'])
    
    # 3. Standardize total_sqft
    df['total_sqft'] = df['total_sqft'].apply(parse_total_sqft)
    
    # 4. Standardize locality
    df['site_location'] = df['site_location'].apply(clean_locality)
    
    # 5. Drop rows with missing critical features
    df = df.dropna(subset=['bhk', 'total_sqft', 'bath', 'price', 'site_location']).copy()
    
    # 6. Compute price_per_sqft (in INR per sqft) strictly for EDA and outlier filtering
    # Note: price is in Lakhs INR (1 Lakh = 100,000 INR)
    df['price_per_sqft'] = (df['price'] * 100000.0) / df['total_sqft']
    
    # 7. Domain outlier filtering
    df = remove_bhk_outliers(df)
    df = remove_bath_outliers(df)
    df = remove_pps_outliers_by_location(df)
    
    # 8. Dimensionality reduction on locality: group locations with < 10 records as 'other'
    location_counts = df['site_location'].value_counts()
    locations_less_than_10 = location_counts[location_counts < 10].index
    df['site_location'] = df['site_location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)
    
    # 9. Save clean processed dataset separately
    if processed_csv_path:
        df.to_csv(processed_csv_path, index=False)
        print(f"Cleaned dataset saved successfully to: {processed_csv_path}")
        print(f"Cleaned dataset shape: {df.shape}")
        
    return df

if __name__ == "__main__":
    clean_housing_data("data/raw/Pune house data.csv", "data/processed/pune_house_data_cleaned.csv")
