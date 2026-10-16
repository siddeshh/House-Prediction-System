import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

from features import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from evaluate import evaluate_predictions, print_evaluation_report

def train_and_evaluate_baselines(data_path: str = "data/processed/pune_house_data_cleaned.csv",
                                 reports_dir: str = "reports",
                                 models_dir: str = "models"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 65)
    print("PHASE 3: TRAINING & EVALUATING BASELINE REGRESSION MODELS")
    print("=" * 65)
    
    # 1. Load Cleaned Dataset
    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    # 2. Defensible Train / Test Split (80% Train, 20% Held-Out Test)
    # Random split is appropriate here as the dataset has no temporal/time-series dimension
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n1. Split Complete:")
    print(f"   - Training Set: {X_train.shape[0]:,} samples")
    print(f"   - Test Set:     {X_test.shape[0]:,} samples")
    
    # 3. Define Baseline Model Candidates
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso Regression (alpha=0.1)": Lasso(alpha=0.1)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    test_summary = {}
    fitted_pipelines = {}
    
    # 4. 5-Fold Cross-Validation on Training Split Only (No Leakage)
    print("\n2. Running 5-Fold Cross Validation on Training Split:")
    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', get_preprocessor()),
            ('regressor', model)
        ])
        
        scores = cross_validate(
            pipe, X_train, y_train, cv=kf,
            scoring={
                'neg_mae': 'neg_mean_absolute_error',
                'neg_rmse': 'neg_root_mean_squared_error',
                'r2': 'r2'
            },
            n_jobs=-1
        )
        
        cv_mae = -scores['test_neg_mae'].mean()
        cv_rmse = -scores['test_neg_rmse'].mean()
        cv_r2 = scores['test_r2'].mean()
        
        cv_summary[name] = {
            "CV_MAE": round(float(cv_mae), 4),
            "CV_RMSE": round(float(cv_rmse), 4),
            "CV_R2": round(float(cv_r2), 4)
        }
        
        print(f"   * {name:30s} -> CV R2: {cv_r2:.4f} | CV MAE: INR {cv_mae:.2f}L | CV RMSE: INR {cv_rmse:.2f}L")
        
        # Fit on entire training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # 5. Evaluate on Held-Out Test Set
        y_test_pred = pipe.predict(X_test)
        test_metrics = evaluate_predictions(y_test.values, y_test_pred)
        test_summary[name] = test_metrics
        
    print("\n3. Final Held-Out Test Set Performance:")
    for name, metrics in test_summary.items():
        print_evaluation_report(name, metrics)
        
    # 6. Select Best Baseline Model by Lowest Test RMSE
    best_name = min(test_summary, key=lambda k: test_summary[k]['RMSE'])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n=> Best Baseline Model: {best_name} (Test RMSE: INR {test_summary[best_name]['RMSE']} Lakhs, R2: {test_summary[best_name]['R2']})")
    
    # 7. Save Best Model Artifact
    model_save_path = os.path.join(models_dir, "baseline_model.joblib")
    joblib.dump(best_pipe, model_save_path)
    print(f"Saved best baseline pipeline to: {model_save_path}")
    
    # 8. Save Metrics Report
    report_data = {
        "dataset": data_path,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cross_validation_5fold": cv_summary,
        "test_evaluation": test_summary,
        "selected_baseline": best_name
    }
    report_path = os.path.join(reports_dir, "baseline_models_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved evaluation metrics report to: {report_path}")
    
    return report_data

if __name__ == "__main__":
    train_and_evaluate_baselines()
