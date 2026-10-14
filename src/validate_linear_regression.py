import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for saving plot images
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from linear_regression_scratch import LinearRegressionScratch

def run_phase1_validation():
    print("=" * 65)
    print("PHASE 1: LINEAR REGRESSION FROM SCRATCH VALIDATION")
    print("=" * 65)
    
    # 1. Generate Synthetic Real Estate Data with Known Ground Truth
    # Equation: Price (Lakhs) = 0.08 * Area(sqft) + 12.0 * BHK + 15.0 + Noise
    np.random.seed(42)
    n_samples = 1000
    
    sqft = np.random.uniform(500, 3500, (n_samples, 1))
    bhk = np.random.randint(1, 6, (n_samples, 1)).astype(float)
    noise = np.random.normal(0, 5.0, (n_samples, 1))
    
    # Ground truth formula
    price = 0.08 * sqft + 12.0 * bhk + 15.0 + noise
    
    X_raw = np.hstack([sqft, bhk])
    y_raw = price
    
    print(f"\n1. Generated {n_samples} synthetic housing records:")
    print(f"   Ground Truth Formula: Price = 0.08 * SqFt + 12.0 * BHK + 15.0")
    print(f"   SqFt Range: [{sqft.min():.1f}, {sqft.max():.1f}] | Mean: {sqft.mean():.1f}")
    print(f"   BHK Range:  [{bhk.min():.0f}, {bhk.max():.0f}] | Mean: {bhk.mean():.1f}")
    print(f"   Price Range: [{price.min():.2f}, {price.max():.2f}] Lakhs")
    
    # 2. Feature Standardization (Z-score normalization)
    # Z = (X - mu) / sigma
    mu = np.mean(X_raw, axis=0)
    sigma = np.std(X_raw, axis=0)
    X_scaled = (X_raw - mu) / sigma
    
    print("\n2. Feature Standardization applied:")
    print(f"   mu: {mu} | sigma: {sigma}")
    print(f"   Scaled X Mean: {np.mean(X_scaled, axis=0).round(4)}")
    print(f"   Scaled X Std:  {np.std(X_scaled, axis=0).round(4)}")
    
    # 3. Model 1: Closed-form Solution (Normal Equation)
    model_normal = LinearRegressionScratch()
    model_normal.fit_normal_equation(X_scaled, y_raw)
    y_pred_normal = model_normal.predict(X_scaled)
    mse_normal = mean_squared_error(y_raw, y_pred_normal)
    r2_normal = r2_score(y_raw, y_pred_normal)
    
    print("\n3. Method 1 — Normal Equation Results:")
    print(f"   Weights (scaled): {model_normal.weights.ravel().round(4)}")
    print(f"   Bias:             {model_normal.bias:.4f}")
    print(f"   Train MSE:        {mse_normal:.4f}")
    print(f"   Train R2:         {r2_normal:.4f}")
    
    # 4. Model 2: Iterative Solution (Gradient Descent)
    learning_rate = 0.05
    n_iterations = 800
    model_gd = LinearRegressionScratch(learning_rate=learning_rate, n_iterations=n_iterations)
    model_gd.fit_gradient_descent(X_scaled, y_raw, verbose=False)
    y_pred_gd = model_gd.predict(X_scaled)
    mse_gd = mean_squared_error(y_raw, y_pred_gd)
    r2_gd = r2_score(y_raw, y_pred_gd)
    
    print(f"\n4. Method 2 — Gradient Descent Results (lr={learning_rate}, iter={n_iterations}):")
    print(f"   Weights (scaled): {model_gd.weights.ravel().round(4)}")
    print(f"   Bias:             {model_gd.bias:.4f}")
    print(f"   Train MSE:        {mse_gd:.4f}")
    print(f"   Train R2:         {r2_gd:.4f}")
    print(f"   Initial Cost:     {model_gd.cost_history[0]:.4f}")
    print(f"   Final Cost:       {model_gd.cost_history[-1]:.4f}")
    
    # 5. Model 3: Scikit-learn Baseline (Identical Scaled Data)
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_scaled, y_raw)
    y_pred_sk = sklearn_model.predict(X_scaled)
    mse_sk = mean_squared_error(y_raw, y_pred_sk)
    r2_sk = r2_score(y_raw, y_pred_sk)
    
    print("\n5. Method 3 — Scikit-Learn LinearRegression Baseline:")
    print(f"   Weights (scaled): {sklearn_model.coef_.ravel().round(4)}")
    print(f"   Bias:             {float(np.ravel(sklearn_model.intercept_)[0]):.4f}")
    print(f"   Train MSE:        {mse_sk:.4f}")
    print(f"   Train R2:         {r2_sk:.4f}")
    
    # 6. Mathematical Equivalence Verification
    print("\n" + "=" * 65)
    print("MATHEMATICAL COMPARISON & TOLERANCE CHECK")
    print("=" * 65)
    weight_diff = np.max(np.abs(model_gd.weights.ravel() - sklearn_model.coef_.ravel()))
    bias_diff = abs(model_gd.bias - float(np.ravel(sklearn_model.intercept_)[0]))
    pred_diff = np.max(np.abs(y_pred_gd - y_pred_sk))
    
    print(f"Max Weight Difference (Scratch GD vs Sklearn): {weight_diff:.6f}")
    print(f"Max Bias Difference   (Scratch GD vs Sklearn): {bias_diff:.6f}")
    print(f"Max Prediction Difference                    : {pred_diff:.6f}")
    
    assert weight_diff < 0.05, f"Weights diverged! Diff: {weight_diff}"
    assert bias_diff < 0.05, f"Bias diverged! Diff: {bias_diff}"
    print("SUCCESS: Scratch implementation matches Scikit-Learn within tolerance!")
    
    # 7. Generate Loss Curve Plot
    import os
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(model_gd.cost_history, color='#2563eb', linewidth=2, label='MSE Cost J(w,b)')
    plt.title('Phase 1: Gradient Descent Loss Convergence Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration Number', fontsize=10)
    plt.ylabel('Cost J(w, b)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = "reports/figures/loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n6. Convergence plot saved to: {plot_path}")

if __name__ == "__main__":
    run_phase1_validation()
