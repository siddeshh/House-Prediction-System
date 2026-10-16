from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes key real-estate evaluation metrics:
    - MAE: Mean Absolute Error in Lakhs INR
    - RMSE: Root Mean Squared Error in Lakhs INR
    - R2: Coefficient of Determination
    - MedAE: Median Absolute Error in Lakhs INR (robust to extreme luxury outliers)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4)
    }

def print_evaluation_report(model_name: str, metrics: Dict[str, float]):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"  * MAE:    INR {metrics['MAE']} Lakhs")
    print(f"  * RMSE:   INR {metrics['RMSE']} Lakhs")
    print(f"  * R2:     {metrics['R2']}")
    print(f"  * MedAE:  INR {metrics['MedAE']} Lakhs")
