import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from features import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from evaluate import evaluate_predictions, print_evaluation_report

def train_advanced_models(data_path: str = "data/processed/pune_house_data_cleaned.csv",
                          reports_dir: str = "reports",
                          figures_dir: str = "reports/figures",
                          models_dir: str = "models"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 65)
    print("PHASE 4: ADVANCED TREE ENSEMBLES & HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)
    
    # 1. Load Data & Create Identical Train/Test Split
    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n1. Data Split:")
    print(f"   Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    
    # 2. Define Advanced Models
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    fitted_pipelines = {}
    test_metrics_summary = {}
    
    # 3. Fit Models & Evaluate on Test Set
    print("\n2. Training Non-Linear Ensemble Models:")
    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', get_preprocessor()),
            ('regressor', regressor)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test.values, y_pred)
        test_metrics_summary[name] = metrics
        print_evaluation_report(name, metrics)
        
    # 4. Hyperparameter Tuning for the Top Performer (Gradient Boosting)
    print("\n3. Hyperparameter Tuning on Gradient Boosting via 3-Fold CV:")
    param_dist = {
        'regressor__n_estimators': [100, 150, 200],
        'regressor__learning_rate': [0.05, 0.08, 0.1],
        'regressor__max_depth': [4, 5, 6],
        'regressor__min_samples_split': [2, 5, 10]
    }
    
    gb_pipe = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    search = RandomizedSearchCV(
        gb_pipe, param_distributions=param_dist,
        n_iter=8, cv=3, scoring='neg_root_mean_squared_error',
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    
    tuned_gb = search.best_estimator_
    y_pred_tuned = tuned_gb.predict(X_test)
    tuned_metrics = evaluate_predictions(y_test.values, y_pred_tuned)
    
    print("\n--- Tuned Gradient Boosting (Best Parameters) ---")
    print(f"  Best Params: {search.best_params_}")
    print_evaluation_report("Tuned Gradient Boosting", tuned_metrics)
    test_metrics_summary["Tuned Gradient Boosting"] = tuned_metrics
    fitted_pipelines["Tuned Gradient Boosting"] = tuned_gb
    
    # 5. Select Global Best Model
    best_name = min(test_metrics_summary, key=lambda k: test_metrics_summary[k]['RMSE'])
    best_model = fitted_pipelines[best_name]
    print(f"\n=> Overall Best Model: {best_name} (Test RMSE: INR {test_metrics_summary[best_name]['RMSE']}L, R2: {test_metrics_summary[best_name]['R2']})")
    
    # Save best model
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Saved production best model to: {best_model_path}")
    
    # 6. Residual Analysis & Plots
    print("\n4. Generating Residual Analysis & Diagnostic Visualizations:")
    y_test_arr = y_test.values
    best_pred = best_model.predict(X_test)
    residuals = y_test_arr - best_pred
    
    # Plot 4A: Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_arr, best_pred, alpha=0.35, color='#2563eb', edgecolors='none', s=25)
    max_val = max(y_test_arr.max(), best_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = y_hat)')
    plt.title(f'Actual vs Predicted House Prices ({best_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Price (INR Lakhs)', fontsize=10)
    plt.ylabel('Predicted Price (INR Lakhs)', fontsize=10)
    plt.legend()
    plt.tight_layout()
    actual_pred_plot = os.path.join(figures_dir, "06_actual_vs_predicted.png")
    plt.savefig(actual_pred_plot, dpi=150)
    plt.close()
    
    # Plot 4B: Residual Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color='#059669')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(f'Residual Error Distribution ({best_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Residual Error (Actual - Predicted) [INR Lakhs]', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.tight_layout()
    residual_plot = os.path.join(figures_dir, "07_residual_distribution.png")
    plt.savefig(residual_plot, dpi=150)
    plt.close()
    
    # 7. Price Tier Error Breakdown
    test_df_analysis = X_test.copy()
    test_df_analysis['actual'] = y_test_arr
    test_df_analysis['predicted'] = best_pred
    test_df_analysis['error'] = np.abs(residuals)
    
    # Define price tiers
    bins = [0, 50, 150, np.inf]
    labels = ['Budget (<50L)', 'Mid-Range (50L-150L)', 'Luxury (>150L)']
    test_df_analysis['price_tier'] = pd.cut(test_df_analysis['actual'], bins=bins, labels=labels)
    
    tier_summary = test_df_analysis.groupby('price_tier', observed=False).agg(
        Count=('actual', 'count'),
        Mean_Actual=('actual', 'mean'),
        MAE=('error', 'mean'),
        MedAE=('error', 'median')
    ).round(2).to_dict(orient='index')
    
    print("\n5. Performance Breakdown Across Price Tiers:")
    for tier, stats in tier_summary.items():
        print(f"   * {tier:22s} -> Count: {stats['Count']} | Mean Price: INR {stats['Mean_Actual']}L | MAE: INR {stats['MAE']}L | MedAE: INR {stats['MedAE']}L")
        
    # 8. Save Advanced Report
    advanced_report = {
        "models_evaluated": test_metrics_summary,
        "best_model_name": best_name,
        "price_tier_performance": tier_summary,
        "plots_saved": [actual_pred_plot, residual_plot]
    }
    with open(os.path.join(reports_dir, "advanced_models_report.json"), "w") as f:
        json.dump(advanced_report, f, indent=2)
    print(f"\nSaved advanced models report to: {os.path.join(reports_dir, 'advanced_models_report.json')}")
    
    return advanced_report

if __name__ == "__main__":
    train_advanced_models()
