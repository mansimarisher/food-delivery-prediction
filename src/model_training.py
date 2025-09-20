"""
Model Training Module for Food Delivery Time Prediction

This module contains functions for training linear regression and logistic regression models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def prepare_regression_data(df: pd.DataFrame, target_col: str = 'Delivery_Time', 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for regression modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data prepared for regression:")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def prepare_classification_data(df: pd.DataFrame, target_col: str = 'Delivery_Time',
                               threshold_method: str = 'median', custom_threshold: Optional[float] = None,
                               test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Prepare data for classification modeling by creating binary target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        threshold_method (str): Method to determine threshold ('median', 'mean', 'custom')
        custom_threshold (float): Custom threshold value
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, threshold
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Determine threshold for binary classification
    if threshold_method == 'median':
        threshold = df[target_col].median()
    elif threshold_method == 'mean':
        threshold = df[target_col].mean()
    elif threshold_method == 'custom' and custom_threshold is not None:
        threshold = custom_threshold
    else:
        threshold = df[target_col].median()  # Default to median
    
    # Create binary target (1 for delayed, 0 for fast)
    df_classification = df.copy()
    df_classification['Delivery_Status'] = (df_classification[target_col] > threshold).astype(int)
    
    X = df_classification.drop([target_col, 'Delivery_Status'], axis=1)
    y = df_classification['Delivery_Status']
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data prepared for classification:")
    print(f"Threshold: {threshold:.2f} minutes")
    print(f"Fast deliveries (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)") 
    print(f"Delayed deliveries (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, threshold


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series,
                           normalize_features: bool = True) -> Tuple[LinearRegression, Optional[StandardScaler]]:
    """
    Train a linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        normalize_features (bool): Whether to normalize features
        
    Returns:
        Tuple: Trained model and scaler (if used)
    """
    scaler = None
    X_train_processed = X_train.copy()
    
    if normalize_features:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    
    print("Linear regression model trained successfully!")
    
    # Feature importance (coefficients)
    if hasattr(X_train, 'columns'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(\"\\nTop 10 most important features:\")
        print(feature_importance.head(10))
    
    return model, scaler


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                             normalize_features: bool = True,
                             hyperparameter_tuning: bool = False) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """
    Train a logistic regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        normalize_features (bool): Whether to normalize features
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        Tuple: Trained model and scaler (if used)
    """
    scaler = None
    X_train_processed = X_train.copy()
    
    if normalize_features:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
    
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        # Parameter grid for tuning
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_processed, y_train)
        
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
    else:
        # Train with default parameters
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_processed, y_train)
    
    print("Logistic regression model trained successfully!")
    
    # Feature importance (coefficients)
    if hasattr(X_train, 'columns'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(\"\\nTop 10 most important features:\")
        print(feature_importance.head(10))
    
    return model, scaler


def evaluate_regression_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series,
                             scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        model: Trained regression model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        scaler: Fitted scaler (if used)
        
    Returns:
        Dict: Evaluation metrics
    """
    # Prepare test data
    X_test_processed = X_test.copy()
    if scaler is not None:
        X_test_processed = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(\"\\nRegression Model Evaluation:\")
    print(f\"Mean Squared Error (MSE): {mse:.2f}\")
    print(f\"Root Mean Squared Error (RMSE): {rmse:.2f}\")
    print(f\"Mean Absolute Error (MAE): {mae:.2f}\")
    print(f\"R-squared (R²): {r2:.4f}\")
    
    return metrics


def evaluate_classification_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series,
                                 scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        model: Trained classification model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        scaler: Fitted scaler (if used)
        
    Returns:
        Dict: Evaluation metrics
    """
    # Prepare test data
    X_test_processed = X_test.copy()
    if scaler is not None:
        X_test_processed = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    }
    
    print(\"\\nClassification Model Evaluation:\")
    print(f\"Accuracy: {accuracy:.4f}\")
    print(f\"Precision: {precision:.4f}\")
    print(f\"Recall: {recall:.4f}\")
    print(f\"F1-Score: {f1:.4f}\")
    print(f\"ROC AUC: {auc:.4f}\")
    
    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                        scaler: Optional[StandardScaler] = None, model_type: str = 'regression') -> Dict[str, np.ndarray]:
    """
    Perform cross-validation on the model.
    
    Args:
        model: Model to validate
        X (pd.DataFrame): Features
        y (pd.Series): Targets
        cv_folds (int): Number of CV folds
        scaler: Scaler (if used)
        model_type (str): 'regression' or 'classification'
        
    Returns:
        Dict: Cross-validation scores
    """
    # Prepare data
    X_processed = X.copy()
    if scaler is not None:
        X_processed = scaler.transform(X)
    
    # Choose scoring metric
    if model_type == 'regression':
        scoring_metrics = ['neg_mean_squared_error', 'r2']
    else:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_processed, y, cv=cv_folds, scoring=metric)
        cv_results[metric] = scores
        
        print(f\"\\n{metric.upper()} - CV Results:\")
        print(f\"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})\")
        print(f\"Scores: {scores}\")
    
    return cv_results


def plot_regression_results(y_test: pd.Series, y_pred: np.ndarray, metrics: Dict[str, float]) -> None:
    """
    Plot regression model results.
    
    Args:
        y_test: Actual test targets
        y_pred: Predicted targets
        metrics: Model metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Actual vs Predicted plot
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Delivery Time')
    axes[0].set_ylabel('Predicted Delivery Time')
    axes[0].set_title('Actual vs Predicted')
    axes[0].text(0.05, 0.95, f'R² = {metrics["R2"]:.4f}', transform=axes[0].transAxes)
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Delivery Time')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                               metrics: Dict[str, float]) -> None:
    """
    Plot classification model results.
    
    Args:
        y_test: Actual test targets
        y_pred: Predicted targets
        y_pred_proba: Predicted probabilities
        metrics: Model metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Fast', 'Delayed'], yticklabels=['Fast', 'Delayed'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["AUC"]:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc=\"lower right\")
    
    plt.tight_layout()
    plt.show()


def save_model_and_scaler(model, scaler: Optional[StandardScaler], model_name: str, save_dir: str = '../models') -> None:
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler (if used)
        model_name (str): Name for saved model
        save_dir (str): Directory to save models
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
    joblib.dump(model, model_path)
    print(f\"Model saved to: {model_path}\")
    
    # Save scaler if exists
    if scaler is not None:
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f\"Scaler saved to: {scaler_path}\")


def load_model_and_scaler(model_name: str, load_dir: str = '../models') -> Tuple[Any, Optional[StandardScaler]]:
    """
    Load trained model and scaler from disk.
    
    Args:
        model_name (str): Name of saved model
        load_dir (str): Directory containing models
        
    Returns:
        Tuple: Loaded model and scaler (if exists)
    """
    import os
    
    # Load model
    model_path = os.path.join(load_dir, f'{model_name}_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f\"Model loaded from: {model_path}\")
    else:
        raise FileNotFoundError(f\"Model not found: {model_path}\")
    
    # Load scaler if exists
    scaler_path = os.path.join(load_dir, f'{model_name}_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f\"Scaler loaded from: {scaler_path}\")
    else:
        scaler = None
        print(\"No scaler found\")
    
    return model, scaler


if __name__ == \"__main__\":
    print(\"Model Training Module - Food Delivery Time Prediction\")
    
    # Example usage with sample data
    from data_preprocessing import create_sample_dataset, encode_categorical_variables
    from feature_engineering import add_distance_features, add_time_features
    
    # Create and prepare sample data
    df = create_sample_dataset(1000)
    df = add_distance_features(df)
    df = add_time_features(df)
    df = encode_categorical_variables(df)
    
    print(f\"\\nDataset prepared. Shape: {df.shape}\")
    
    # Train regression model
    print(\"\\n\" + \"=\"*50)
    print(\"TRAINING REGRESSION MODEL\")
    print(\"=\"*50)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_regression_data(df)
    reg_model, reg_scaler = train_linear_regression(X_train_reg, y_train_reg)
    reg_metrics = evaluate_regression_model(reg_model, X_test_reg, y_test_reg, reg_scaler)
    
    # Train classification model
    print(\"\\n\" + \"=\"*50)
    print(\"TRAINING CLASSIFICATION MODEL\")
    print(\"=\"*50)
    
    X_train_cls, X_test_cls, y_train_cls, y_test_cls, threshold = prepare_classification_data(df)
    cls_model, cls_scaler = train_logistic_regression(X_train_cls, y_train_cls)
    cls_metrics = evaluate_classification_model(cls_model, X_test_cls, y_test_cls, cls_scaler)
    
    print(\"\\nModel training completed successfully!\")