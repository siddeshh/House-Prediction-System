# Senior ML Engineer Interview Preparation Guide

This guide covers real-world technical interview questions based on the architecture and implementations of the **Pune House Price Prediction** project.

---

### Q1: Why did you implement Linear Regression from scratch with NumPy before using Scikit-Learn?
**Answer:**
Building the algorithm from first principles demonstrates a mastery of the underlying mathematics rather than treating machine learning as an opaque API. By writing vectorized matrix multiplication (`X @ w + b`), the analytic Normal Equation ($\theta = (X^T X)^{-1} X^T y$), and the calculus of the gradient descent update loop ($\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$), I proved exact parity with Scikit-Learn to within $10^{-6}$ precision on identical data.

---

### Q2: What is the computational difference between the Normal Equation and Gradient Descent?
**Answer:**
- **The Normal Equation** computes the exact global minimum in a single closed-form step without needing a learning rate $\alpha$ or iterative loops. However, it requires inverting an $(n \times n)$ matrix, which has an $\mathcal{O}(n^3)$ computational complexity. When features $n$ scale past several thousand (e.g. after one-hot encoding), matrix inversion becomes computationally prohibitive or numerically singular.
- **Gradient Descent** scales efficiently to millions of samples and high-dimensional feature spaces ($\mathcal{O}(k \cdot m \cdot n)$), but requires careful hyperparameter tuning of the learning rate $\alpha$ and convergence criteria.

---

### Q3: What is the famous NumPy broadcasting gotcha when computing regression errors?
**Answer:**
If the ground truth $y$ has shape `(m,)` (a 1D array) and predictions $\hat{y}$ have shape `(m, 1)` (a 2D column vector), executing `y_hat - y` does **not** subtract element-wise. NumPy broadcasts `(m, 1)` with `(1, m)` into a giant `(m, m)` 2D matrix of pairwise differences! This silently consumes massive RAM and completely corrupts the Mean Squared Error calculation. The fix is strictly ensuring both arrays are 2D column vectors via `.reshape(-1, 1)`.

---

### Q4: Why is Feature Scaling mandatory for Gradient Descent, but not for the Normal Equation?
**Answer:**
When features have disparate scales (e.g., Area is $\approx 2000$ while BHK is $\approx 3$), the loss surface forms an extremely elongated elliptical bowl. The gradient with respect to Area is orders of magnitude larger than for BHK, causing the weights to oscillate wildly or diverge unless an infinitesimally small learning rate is used. Z-score standardization ($z = \frac{x - \mu}{\sigma}$) sphericalizes the loss contours, enabling direct and steady gradient descent steps. In contrast, the Normal Equation solves the system of linear equations directly via matrix projection, making it invariant to linear scaling of the features.

---

### Q5: What is Data Leakage, and how did your pipeline prevent it?
**Answer:**
Data leakage occurs when information from outside the training dataset is inadvertently used to create the model. A classic beginner mistake is applying `StandardScaler` or `SimpleImputer` to the entire dataset before `train_test_split`, which leaks test set means and standard deviations into the training fold.
In my implementation, all preprocessing steps are encapsulated inside a Scikit-Learn `ColumnTransformer` within an end-to-end `Pipeline`. The preprocessor is `.fit()` **strictly on the training split**, and only `.transform()` is executed on validation and test folds.

---

### Q6: Why did you not use `price_per_sqft` as an input feature for model training?
**Answer:**
`price_per_sqft` is calculated directly as $\frac{\text{price} \times 10^5}{\text{total\_sqft}}$. Because `price` is the target variable we are attempting to predict, feeding `price_per_sqft` into the model would be catastrophic **target leakage**. The model would trivially learn that $\text{price} = \frac{\text{price\_per\_sqft} \times \text{total\_sqft}}{10^5}$ and achieve an artificial $R^2 = 1.0$ during training, but completely fail in production where the true price is unknown. We used it strictly for EDA and domain outlier filtering.

---

### Q7: Why did tree ensembles (Random Forest, Gradient Boosting) outperform Linear, Ridge, and Lasso models?
**Answer:**
Linear models assume an additive, planar relationship between features and price: each additional square foot adds a fixed price increment regardless of locality. In reality, real estate pricing exhibits complex **non-linear interaction effects**:
- 1,000 sq.ft. in a prime neighborhood (Koregaon Park) commands a much higher marginal rate than in outer suburbs.
- The value of an extra bathroom depends heavily on whether the property is a compact 1 BHK or a luxury 4 BHK.
Tree ensembles partition the feature space non-linearly. While Ridge and Lasso achieved $R^2 \approx 0.45$, **Tuned Gradient Boosting achieved $R^2 \approx 0.80$**, reducing RMSE by 39%.

---

### Q8: What did your residual analysis show regarding heteroscedasticity?
**Answer:**
Plotting residuals ($y - \hat{y}$) against actual prices revealed **heteroscedasticity** (non-constant variance):
- On Budget homes (<₹50L), the MAE was only ₹10.37 Lakhs.
- On Luxury homes (>₹150L), the MAE increased to ₹46.64 Lakhs.
This occurs because lower-priced apartments are commoditized (priced strictly per square foot), whereas luxury penthouses and bungalows possess subjective value drivers (panoramic views, private terraces, architectural prestige) that are not captured in basic tabular features.

---

### Q9: How does the FastAPI service handle unseen localities and malformed requests?
**Answer:**
1. **Pydantic Validation:** The request schema enforces strict data bounds (e.g. `total_sqft` must be between 150 and 30,000; `bhk` between 1 and 15). Malformed payloads (negative areas, strings in numeric fields) are rejected immediately at the HTTP boundary with **HTTP 422 Unprocessable Entity**.
2. **Unseen Localities:** The trained Scikit-Learn `OneHotEncoder` is configured with `handle_unknown='ignore'`. When a user provides a previously unseen locality, all locality one-hot columns encode as zeros, allowing the model to base its prediction on property dimensions and baseline bias without raising a runtime exception.

---

### Q10: What did your dataset provenance audit uncover, and why does it matter?
**Answer:**
During exploratory inspection, I discovered that the dataset contains exactly 13,320 rows and identical missingness patterns to Kaggle's Bengaluru House Price dataset, with locality names synthetically mapped in round-robin fashion across 97 Pune locations.
In an engineering portfolio, discovering and transparently documenting this limitation distinguishes a professional engineer from a script-runner. I explicitly documented this finding in the README and model metadata: the model serves as an educational ML demonstration, and should not be used for actual Pune market financial appraisals.
