import numpy as np
from typing import Tuple, List, Optional

class LinearRegressionScratch:
    """
    Linear Regression implemented from first principles using pure NumPy.
    Supports both Gradient Descent and the closed-form Normal Equation.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.cost_history: List[float] = []
        
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Solves Linear Regression analytically via the Normal Equation:
        w_all = (X_b.T @ X_b)^(-1) @ X_b.T @ y
        where X_b includes a leading column of 1s for the bias term.
        """
        # Ensure correct shapes
        y = y.reshape(-1, 1)
        m = X.shape[0]
        
        # Add intercept column of ones: X_b has shape (m, n + 1)
        X_b = np.c_[np.ones((m, 1)), X]
        
        # Normal equation computation: (X_b^T @ X_b)^(-1) @ X_b^T @ y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias = float(theta[0, 0])
        self.weights = theta[1:, :]
        
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes Mean Squared Error: J(w, b) = 1/(2m) * sum((Xw + b - y)^2)
        """
        m = X.shape[0]
        y_hat = X @ self.weights + self.bias
        cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
        return float(cost)

    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Trains model weights and bias iteratively using Gradient Descent.
        """
        y = y.reshape(-1, 1)
        m, n = X.shape
        
        # Initialize parameters to zeros
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        self.cost_history = []
        
        for iteration in range(self.n_iterations):
            # 1. Forward pass: predictions
            y_hat = X @ self.weights + self.bias
            
            # 2. Compute error residual
            error = y_hat - y
            
            # 3. Compute gradients
            dw = (1 / m) * (X.T @ error)      # Shape (n, 1)
            db = float((1 / m) * np.sum(error))  # Scalar
            
            # 4. Parameter update rule
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 5. Track cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            if verbose and (iteration % (self.n_iterations // 10) == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration:5d} | Cost (MSE): {cost:.4f} | Bias: {self.bias:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts continuous values for input feature matrix X.
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call fit_gradient_descent or fit_normal_equation.")
        return X @ self.weights + self.bias
