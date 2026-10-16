from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES: List[str] = ['total_sqft', 'bath', 'bhk', 'balcony']
CATEGORICAL_FEATURES: List[str] = ['site_location', 'area_type']
TARGET: str = 'price'

def get_preprocessor() -> ColumnTransformer:
    """
    Builds a robust, leakage-free ColumnTransformer pipeline.
    - Numeric features: Median Imputation -> StandardScaler
    - Categorical features: Constant Imputation -> OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERIC_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor
