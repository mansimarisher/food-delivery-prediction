"""
Configuration Module for Food Delivery Time Prediction Project

This module contains all configuration parameters, model hyperparameters,
and project settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data processing and paths."""
    
    # File paths
    DATA_DIR: str = "../data"
    MODELS_DIR: str = "../models"
    REPORTS_DIR: str = "../reports"
    NOTEBOOKS_DIR: str = "../notebooks"
    
    # Dataset file
    DATASET_FILENAME: str = "Food_Delivery_Time_Prediction.csv"
    
    # Data processing parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Missing value handling
    MISSING_VALUE_STRATEGY: str = "median"  # 'median', 'mean', 'mode'
    
    # Outlier handling
    OUTLIER_METHOD: str = "iqr"  # 'iqr', 'zscore'
    REMOVE_OUTLIERS: bool = False
    
    # Feature encoding
    ENCODING_METHOD: str = "onehot"  # 'onehot', 'label'
    
    @property
    def dataset_path(self) -> str:
        """Full path to the dataset file."""
        return os.path.join(self.DATA_DIR, self.DATASET_FILENAME)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Distance calculation
    USE_HAVERSINE_DISTANCE: bool = True
    DISTANCE_CATEGORIES: List[float] = None
    
    # Time-based features
    CREATE_TIME_FEATURES: bool = True
    RUSH_HOUR_RANGES: List[tuple] = None
    
    # Feature scaling
    SCALING_METHOD: str = "standard"  # 'standard', 'minmax', 'robust'
    
    # Polynomial features
    USE_POLYNOMIAL_FEATURES: bool = False
    POLYNOMIAL_DEGREE: int = 2
    
    # Feature selection
    USE_CORRELATION_FILTER: bool = False
    CORRELATION_THRESHOLD: float = 0.01
    
    def __post_init__(self):
        if self.DISTANCE_CATEGORIES is None:
            self.DISTANCE_CATEGORIES = [0, 2, 5, 10, float('inf')]
        
        if self.RUSH_HOUR_RANGES is None:
            self.RUSH_HOUR_RANGES = [(11, 13), (18, 20)]


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    
    # General model settings
    TARGET_COLUMN: str = "Delivery_Time"
    NORMALIZE_FEATURES: bool = True
    CROSS_VALIDATION_FOLDS: int = 5
    
    # Linear Regression parameters
    LINEAR_REGRESSION_PARAMS: Dict[str, Any] = None
    
    # Logistic Regression parameters
    LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = None
    LOGISTIC_THRESHOLD_METHOD: str = "median"  # 'median', 'mean', 'custom'
    LOGISTIC_CUSTOM_THRESHOLD: Optional[float] = None
    
    # Hyperparameter tuning
    USE_HYPERPARAMETER_TUNING: bool = False
    HYPERPARAMETER_CV_FOLDS: int = 5
    
    # Model evaluation
    EVALUATION_METRICS_REGRESSION: List[str] = None
    EVALUATION_METRICS_CLASSIFICATION: List[str] = None
    
    # Model saving
    SAVE_MODELS: bool = True
    MODEL_SAVE_FORMAT: str = "pickle"  # 'pickle', 'joblib'
    
    def __post_init__(self):
        if self.LINEAR_REGRESSION_PARAMS is None:
            self.LINEAR_REGRESSION_PARAMS = {
                'fit_intercept': True,
                'normalize': False  # Deprecated in newer sklearn versions
            }
        
        if self.LOGISTIC_REGRESSION_PARAMS is None:
            self.LOGISTIC_REGRESSION_PARAMS = {
                'random_state': 42,
                'max_iter': 1000,
                'solver': 'lbfgs'
            }
        
        if self.EVALUATION_METRICS_REGRESSION is None:
            self.EVALUATION_METRICS_REGRESSION = ['MSE', 'RMSE', 'MAE', 'R2']
        
        if self.EVALUATION_METRICS_CLASSIFICATION is None:
            self.EVALUATION_METRICS_CLASSIFICATION = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']


@dataclass
class VisualizationConfig:
    """Configuration for plots and visualizations."""
    
    # General plot settings
    FIGURE_SIZE: tuple = (12, 8)
    PLOT_STYLE: str = "seaborn-v0_8"  # matplotlib style
    COLOR_PALETTE: str = "husl"  # seaborn color palette
    
    # Plot-specific settings
    CORRELATION_HEATMAP_CMAP: str = "coolwarm"
    SCATTER_ALPHA: float = 0.6
    HIST_BINS: int = 30
    
    # Save settings
    SAVE_PLOTS: bool = True
    PLOT_DPI: int = 300
    PLOT_FORMAT: str = "png"  # 'png', 'pdf', 'svg'


@dataclass 
class ExperimentConfig:
    """Configuration for experiment tracking and reproducibility."""
    
    # Experiment metadata
    PROJECT_NAME: str = "Food Delivery Time Prediction"
    EXPERIMENT_NAME: str = "Linear_Logistic_Regression_Baseline"
    VERSION: str = "1.0.0"
    
    # Reproducibility
    RANDOM_SEED: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Results tracking
    TRACK_EXPERIMENTS: bool = True
    SAVE_EXPERIMENT_LOGS: bool = True


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.visualization = VisualizationConfig()
        self.experiment = ExperimentConfig()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert config to dictionary format."""
        return {
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'visualization': self.visualization.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create new config instance and update with loaded values
        config = cls()
        for section_name, section_config in config_dict.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_config.items():
                    setattr(section, key, value)
        
        print(f"Configuration loaded from: {filepath}")
        return config


# Default configuration instance
default_config = Config()

# Hyperparameter grids for tuning
HYPERPARAMETER_GRIDS = {
    'linear_regression': {
        # Linear regression doesn't have many hyperparameters to tune
        'fit_intercept': [True, False]
    },
    
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [500, 1000, 2000]
    }
}

# Feature importance thresholds
FEATURE_IMPORTANCE_THRESHOLDS = {
    'correlation_threshold': 0.01,
    'variance_threshold': 0.01,
    'mutual_info_threshold': 0.01
}

# Data validation rules
DATA_VALIDATION_RULES = {
    'required_columns': [
        'Customer_Lat', 'Customer_Lng', 'Restaurant_Lat', 'Restaurant_Lng',
        'Weather_Conditions', 'Traffic_Conditions', 'Vehicle_Type',
        'Delivery_Time'
    ],
    'numeric_columns': [
        'Customer_Lat', 'Customer_Lng', 'Restaurant_Lat', 'Restaurant_Lng',
        'Delivery_Time', 'Order_Cost'
    ],
    'categorical_columns': [
        'Weather_Conditions', 'Traffic_Conditions', 'Vehicle_Type'
    ],
    'value_ranges': {
        'Delivery_Time': (0, 300),  # 0 to 5 hours in minutes
        'Order_Cost': (0, 1000),    # $0 to $1000
        'Customer_Lat': (25, 50),   # Reasonable latitude range
        'Customer_Lng': (-130, -65) # Reasonable longitude range for US
    }
}

# Model performance benchmarks
PERFORMANCE_BENCHMARKS = {
    'regression': {
        'excellent_r2': 0.8,
        'good_r2': 0.6,
        'acceptable_mae': 10.0  # minutes
    },
    'classification': {
        'excellent_accuracy': 0.85,
        'good_accuracy': 0.75,
        'excellent_f1': 0.80
    }
}


if __name__ == "__main__":
    # Example usage
    print("Model Configuration Module")
    print("=" * 40)
    
    # Create default config
    config = Config()
    
    # Display configuration
    print(f"Project: {config.experiment.PROJECT_NAME}")
    print(f"Target Column: {config.model.TARGET_COLUMN}")
    print(f"Test Size: {config.data.TEST_SIZE}")
    print(f"Random State: {config.data.RANDOM_STATE}")
    print(f"Scaling Method: {config.features.SCALING_METHOD}")
    
    # Save configuration to file
    config.save_config("../config/current_config.json")
    
    print("\nConfiguration setup completed!")