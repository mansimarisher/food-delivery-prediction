"""
Utility Functions for Food Delivery Time Prediction Project

This module contains helper functions for visualization, reporting,
model evaluation, and general utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def setup_plotting_style(style: str = 'seaborn-v0_8', palette: str = 'husl', 
                        figure_size: tuple = (12, 8)) -> None:
    """
    Set up consistent plotting style for all visualizations.
    
    Args:
        style (str): Matplotlib style to use
        palette (str): Seaborn color palette
        figure_size (tuple): Default figure size
    """
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    
    sns.set_palette(palette)
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10


def save_plot(filename: str, save_dir: str = '../reports', 
              dpi: int = 300, format: str = 'png', bbox_inches: str = 'tight') -> str:
    """
    Save plot with consistent formatting.
    
    Args:
        filename (str): Name of the file
        save_dir (str): Directory to save the plot
        dpi (int): Resolution in dots per inch
        format (str): File format (png, pdf, svg)
        bbox_inches (str): Bounding box format
        
    Returns:
        str: Full path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"{filename}.{format}")
    plt.savefig(full_path, dpi=dpi, format=format, bbox_inches=bbox_inches)
    return full_path


def create_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 10), 
                              save_path: Optional[str] = None) -> None:
    """
    Create and save correlation heatmap.
    
    Args:
        df (pd.DataFrame): Dataframe with numerical columns
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the plot
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        save_plot('correlation_heatmap', save_path)
    
    plt.show()


def plot_feature_importance(feature_names: List[str], importance_scores: List[float],
                           title: str = 'Feature Importance', top_n: int = 15,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_names (List[str]): Names of features
        importance_scores (List[float]): Importance scores
        title (str): Plot title
        top_n (int): Number of top features to show
        save_path (str, optional): Path to save the plot
    """
    # Create dataframe and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importance_scores)
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_plot('feature_importance', save_path)
    
    plt.show()


def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = 'Model', save_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted values for regression.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(f'{model_name.lower()}_predictions', save_path)
    
    plt.show()


def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray, class_labels: List[str] = ['Fast', 'Delayed'],
                               model_name: str = 'Model', save_path: Optional[str] = None) -> None:
    """
    Plot classification results including confusion matrix and ROC curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities
        class_labels (List[str]): Class labels
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[0].set_title(f'{model_name}: Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_name}: ROC Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(f'{model_name.lower()}_classification', save_path)
    
    plt.show()


def plot_distribution_comparison(data: Dict[str, np.ndarray], title: str = 'Distribution Comparison',
                                save_path: Optional[str] = None) -> None:
    """
    Plot distribution comparison for multiple datasets.
    
    Args:
        data (Dict[str, np.ndarray]): Dictionary with dataset names and values
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in data.items():
        plt.hist(values, bins=30, alpha=0.7, label=name, density=True)
    
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_plot('distribution_comparison', save_path)
    
    plt.show()


def generate_model_report(model_metrics: Dict[str, float], model_name: str,
                         feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate comprehensive model evaluation report.
    
    Args:
        model_metrics (Dict[str, float]): Model performance metrics
        model_name (str): Name of the model
        feature_importance (pd.DataFrame, optional): Feature importance data
        
    Returns:
        Dict[str, Any]: Complete model report
    """
    report = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': model_metrics,
        'evaluation_summary': {}
    }
    
    # Add performance interpretation
    if 'R2' in model_metrics:
        r2_score = model_metrics['R2']
        if r2_score > 0.8:
            report['evaluation_summary']['r2_interpretation'] = 'Excellent'
        elif r2_score > 0.6:
            report['evaluation_summary']['r2_interpretation'] = 'Good'
        else:
            report['evaluation_summary']['r2_interpretation'] = 'Needs Improvement'
    
    if 'Accuracy' in model_metrics:
        accuracy = model_metrics['Accuracy']
        if accuracy > 0.85:
            report['evaluation_summary']['accuracy_interpretation'] = 'Excellent'
        elif accuracy > 0.75:
            report['evaluation_summary']['accuracy_interpretation'] = 'Good'
        else:
            report['evaluation_summary']['accuracy_interpretation'] = 'Needs Improvement'
    
    # Add feature importance if provided
    if feature_importance is not None:
        report['feature_importance'] = {
            'top_features': feature_importance.head(10).to_dict('records'),
            'total_features': len(feature_importance)
        }
    
    return report


def save_experiment_results(results: Dict[str, Any], filename: str, 
                           save_dir: str = '../reports') -> str:
    """
    Save experiment results to JSON file.
    
    Args:
        results (Dict[str, Any]): Results to save
        filename (str): Name of the file (without extension)
        save_dir (str): Directory to save the file
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filename = f"{timestamp}_{filename}.json"
    full_path = os.path.join(save_dir, full_filename)
    
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {full_path}")
    return full_path


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from: {filepath}")
    return results


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              cost_per_minute: float = 0.5) -> Dict[str, float]:
    """
    Calculate business-relevant metrics from predictions.
    
    Args:
        y_true (np.ndarray): Actual delivery times
        y_pred (np.ndarray): Predicted delivery times  
        cost_per_minute (float): Cost per minute of delivery delay
        
    Returns:
        Dict[str, float]: Business metrics
    """
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    
    metrics = {
        'mean_prediction_error': np.mean(errors),
        'mean_absolute_error_minutes': np.mean(absolute_errors),
        'total_time_error_hours': np.sum(absolute_errors) / 60,
        'estimated_cost_impact': np.sum(absolute_errors) * cost_per_minute,
        'accuracy_within_5_minutes': np.mean(absolute_errors <= 5) * 100,
        'accuracy_within_10_minutes': np.mean(absolute_errors <= 10) * 100
    }
    
    return metrics


def create_executive_summary(model_results: Dict[str, Dict[str, float]],
                           business_metrics: Dict[str, float]) -> str:
    """
    Create executive summary text from model results.
    
    Args:
        model_results (Dict[str, Dict[str, float]]): Model performance results
        business_metrics (Dict[str, float]): Business impact metrics
        
    Returns:
        str: Executive summary text
    """
    summary = f"""
# Food Delivery Time Prediction - Executive Summary

## Project Overview
This analysis developed machine learning models to predict food delivery times and classify deliveries as fast or delayed.

## Key Findings

### Model Performance
"""
    
    # Add regression results if available
    if 'linear_regression' in model_results:
        lr_metrics = model_results['linear_regression']
        r2_score = lr_metrics.get('R2', 0)
        mae = lr_metrics.get('MAE', 0)
        
        summary += f"""
**Delivery Time Prediction (Linear Regression):**
- Model explains {r2_score*100:.1f}% of delivery time variation
- Average prediction error: {mae:.1f} minutes
"""
    
    # Add classification results if available
    if 'logistic_regression' in model_results:
        cls_metrics = model_results['logistic_regression']
        accuracy = cls_metrics.get('Accuracy', 0)
        f1_score = cls_metrics.get('F1', 0)
        
        summary += f"""
**Fast/Delayed Classification (Logistic Regression):**
- Classification accuracy: {accuracy*100:.1f}%
- F1-score: {f1_score:.3f}
"""
    
    # Add business impact
    if business_metrics:
        summary += f"""
### Business Impact
- Average prediction accuracy within 5 minutes: {business_metrics.get('accuracy_within_5_minutes', 0):.1f}%
- Estimated cost impact of prediction errors: ${business_metrics.get('estimated_cost_impact', 0):.0f}
- Total time variance: {business_metrics.get('total_time_error_hours', 0):.1f} hours
"""
    
    summary += """
## Recommendations
1. **Operational**: Optimize delivery routes using distance and traffic predictions
2. **Staffing**: Increase personnel during identified rush hours (11-13h, 18-20h)  
3. **Technology**: Implement real-time traffic and weather monitoring
4. **Customer Service**: Provide proactive delay notifications based on model predictions

## Next Steps
- Deploy models in production environment
- Set up monitoring for model performance drift
- Collect additional features (real-time traffic, weather data)
- Implement A/B testing for model-driven optimizations
"""
    
    return summary


def validate_model_inputs(X: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that input data has required columns for model prediction.
    
    Args:
        X (pd.DataFrame): Input feature data
        required_columns (List[str]): Required column names
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in X.columns]
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns


def print_model_summary(model_name: str, metrics: Dict[str, float]) -> None:
    """
    Print formatted model performance summary.
    
    Args:
        model_name (str): Name of the model
        metrics (Dict[str, float]): Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:20}: {value:.4f}")
        else:
            print(f"{metric_name:20}: {value}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage and testing
    print("Utility Functions Module - Food Delivery Time Prediction")
    print("=" * 60)
    
    # Test plotting setup
    setup_plotting_style()
    print("✓ Plotting style configured")
    
    # Test sample data generation for visualization
    np.random.seed(42)
    sample_data = {
        'Actual': np.random.normal(35, 10, 100),
        'Predicted': np.random.normal(35, 10, 100) + np.random.normal(0, 5, 100)
    }
    
    # Test metrics calculation
    business_metrics = calculate_business_metrics(
        sample_data['Actual'], 
        sample_data['Predicted']
    )
    
    print("✓ Business metrics calculated:")
    for metric, value in business_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    print("\n✓ Utility functions module loaded successfully!")
    print("All visualization and reporting functions are ready to use.")