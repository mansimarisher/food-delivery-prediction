"""
Data Preprocessing Module for Food Delivery Time Prediction

This module contains functions for loading, cleaning, and preprocessing
the food delivery dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the food delivery dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Dataset not found at {file_path}")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()


def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample dataset
    """
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'Customer_Lat': np.random.uniform(40.7, 40.8, n_samples),
        'Customer_Lng': np.random.uniform(-74.0, -73.9, n_samples),
        'Restaurant_Lat': np.random.uniform(40.7, 40.8, n_samples),
        'Restaurant_Lng': np.random.uniform(-74.0, -73.9, n_samples),
        'Weather_Conditions': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n_samples),
        'Traffic_Conditions': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Vehicle_Type': np.random.choice(['Bike', 'Scooter', 'Car'], n_samples),
        'Delivery_Person_Experience': np.random.randint(1, 10, n_samples),
        'Order_Cost': np.random.uniform(10, 100, n_samples),
        'Order_Priority': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Delivery_Time': np.random.uniform(15, 60, n_samples)
    }
    
    # Add some realistic relationships
    df = pd.DataFrame(data)
    
    # Make delivery time correlate with traffic and distance
    traffic_multiplier = df['Traffic_Conditions'].map({'Low': 0.8, 'Medium': 1.0, 'High': 1.3})
    weather_multiplier = df['Weather_Conditions'].map({'Sunny': 1.0, 'Cloudy': 1.1, 'Rainy': 1.2})
    
    df['Delivery_Time'] = df['Delivery_Time'] * traffic_multiplier * weather_multiplier
    
    print(f"Sample dataset created with {n_samples} records")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Missing values count per column
    """
    missing_values = df.isnull().sum()
    print("Missing Values Summary:")
    print(missing_values[missing_values > 0])
    
    if missing_values.sum() == 0:
        print("No missing values found!")
    else:
        print(f"Total missing values: {missing_values.sum()}")
        print(f"Percentage of missing data: {(missing_values.sum() / len(df)) * 100:.2f}%")
    
    return missing_values


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('median', 'mean', 'mode')
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mean':
                fill_value = df_clean[col].mean()
            else:
                fill_value = df_clean[col].median()
            
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"Filled {col} with {strategy}: {fill_value:.2f}")
    
    # Categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} with mode: {mode_val}")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> Tuple[pd.DataFrame, float, float]:
    """
    Detect outliers in a specific column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        method (str): Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        Tuple[pd.DataFrame, float, float]: Outliers dataframe, lower bound, upper bound
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        outliers = df[z_scores > 3]
        lower_bound = df[column].mean() - 3 * df[column].std()
        upper_bound = df[column].mean() + 3 * df[column].std()
    
    return outliers, lower_bound, upper_bound


def remove_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to remove outliers from
        method (str): Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    df_clean = df.copy()
    initial_size = len(df_clean)
    
    for column in columns:
        outliers, lower_bound, upper_bound = detect_outliers(df_clean, column, method)
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        print(f"Removed {len(outliers)} outliers from {column}")
    
    final_size = len(df_clean)
    print(f"Dataset size reduced from {initial_size} to {final_size} rows")
    print(f"Removed {initial_size - final_size} total outliers ({((initial_size - final_size) / initial_size) * 100:.2f}%)")
    
    return df_clean


def encode_categorical_variables(df: pd.DataFrame, encoding_type: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoding_type (str): Type of encoding ('onehot', 'label')
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("No categorical columns found for encoding")
        return df_encoded
    
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    
    if encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        print(f"One-hot encoding applied. New shape: {df_encoded.shape}")
    
    elif encoding_type == 'label':
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            print(f"Label encoding applied to {col}")
    
    return df_encoded


def get_dataset_info(df: pd.DataFrame) -> None:
    """
    Print comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Information:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module - Food Delivery Time Prediction")
    
    # Load or create sample data
    df = load_dataset("../data/Food_Delivery_Time_Prediction.csv")
    
    # Get dataset information
    get_dataset_info(df)
    
    # Check for missing values
    missing_vals = check_missing_values(df)
    
    # Handle missing values if any
    df_clean = handle_missing_values(df)
    
    # Check for outliers in delivery time
    outliers, lower, upper = detect_outliers(df_clean, 'Delivery_Time')
    print(f"\\nOutliers in Delivery_Time: {len(outliers)} ({(len(outliers)/len(df_clean)*100):.1f}%)")
    
    # Encode categorical variables
    df_encoded = encode_categorical_variables(df_clean)
    
    print(f"\\nFinal processed dataset shape: {df_encoded.shape}")
    print("Data preprocessing completed successfully!")