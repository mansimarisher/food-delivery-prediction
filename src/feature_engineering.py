"""
Feature Engineering Module for Food Delivery Time Prediction

This module contains functions for creating new features from raw data,
including distance calculations, time-based features, and feature scaling.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def calculate_haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lng1: Latitude and longitude of first point
        lat2, lng2: Latitude and longitude of second point
        
    Returns:
        float: Distance in kilometers
    """
    return geodesic((lat1, lng1), (lat2, lng2)).kilometers


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance-related features to the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe with location coordinates
        
    Returns:
        pd.DataFrame: Dataframe with added distance features
    """
    df_features = df.copy()
    
    # Check if coordinate columns exist
    coord_cols = ['Customer_Lat', 'Customer_Lng', 'Restaurant_Lat', 'Restaurant_Lng']
    if all(col in df_features.columns for col in coord_cols):
        print("Calculating distance features...")
        
        # Calculate direct distance
        df_features['Distance_km'] = df_features.apply(
            lambda row: calculate_haversine_distance(
                row['Customer_Lat'], row['Customer_Lng'],
                row['Restaurant_Lat'], row['Restaurant_Lng']
            ), axis=1
        )
        
        # Create distance categories
        df_features['Distance_Category'] = pd.cut(
            df_features['Distance_km'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Very Close', 'Close', 'Medium', 'Far']
        )
        
        # Calculate coordinate differences (can indicate direction)
        df_features['Lat_Diff'] = df_features['Restaurant_Lat'] - df_features['Customer_Lat']
        df_features['Lng_Diff'] = df_features['Restaurant_Lng'] - df_features['Customer_Lng']
        
        print(f"Added distance features. Mean distance: {df_features['Distance_km'].mean():.2f} km")
        
    else:
        print("Coordinate columns not found. Creating synthetic distance feature...")
        # Create synthetic distance for demonstration
        np.random.seed(42)
        df_features['Distance_km'] = np.random.uniform(1, 20, len(df_features))
        df_features['Distance_Category'] = pd.cut(
            df_features['Distance_km'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Very Close', 'Close', 'Medium', 'Far']
        )
    
    return df_features


def add_time_features(df: pd.DataFrame, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add time-based features to the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_col (str, optional): Name of timestamp column if exists
        
    Returns:
        pd.DataFrame: Dataframe with added time features
    """
    df_features = df.copy()
    
    if timestamp_col and timestamp_col in df_features.columns:
        print(f"Creating time features from {timestamp_col}...")
        
        # Convert to datetime if not already
        df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
        
        # Extract time components
        df_features['Hour'] = df_features[timestamp_col].dt.hour
        df_features['Day_of_Week'] = df_features[timestamp_col].dt.dayofweek
        df_features['Month'] = df_features[timestamp_col].dt.month
        df_features['Is_Weekend'] = (df_features['Day_of_Week'] >= 5).astype(int)
        
    else:
        print("Creating synthetic time features...")
        # Create synthetic time features for demonstration
        np.random.seed(42)
        df_features['Hour'] = np.random.randint(0, 24, len(df_features))
        df_features['Day_of_Week'] = np.random.randint(0, 7, len(df_features))
        df_features['Month'] = np.random.randint(1, 13, len(df_features))
        df_features['Is_Weekend'] = (df_features['Day_of_Week'] >= 5).astype(int)
    
    # Create rush hour features
    df_features['Is_Rush_Hour'] = df_features['Hour'].apply(
        lambda x: 1 if (11 <= x <= 13) or (18 <= x <= 20) else 0
    )
    
    # Create meal time features
    df_features['Meal_Time'] = df_features['Hour'].apply(classify_meal_time)
    
    # Create time of day categories
    df_features['Time_of_Day'] = df_features['Hour'].apply(classify_time_of_day)
    
    print("Time features added successfully")
    return df_features


def classify_meal_time(hour: int) -> str:
    """
    Classify hour into meal time categories.
    
    Args:
        hour (int): Hour of the day (0-23)
        
    Returns:
        str: Meal time category
    """
    if 6 <= hour < 10:
        return 'Breakfast'
    elif 11 <= hour < 15:
        return 'Lunch'
    elif 17 <= hour < 21:
        return 'Dinner'
    else:
        return 'Other'


def classify_time_of_day(hour: int) -> str:
    """
    Classify hour into time of day categories.
    
    Args:
        hour (int): Hour of the day (0-23)
        
    Returns:
        str: Time of day category
    """
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    df_features = df.copy()
    
    print("Creating interaction features...")
    
    # Distance and traffic interaction
    if 'Distance_km' in df_features.columns and 'Traffic_Conditions_High' in df_features.columns:
        df_features['Distance_x_HighTraffic'] = (
            df_features['Distance_km'] * df_features['Traffic_Conditions_High']
        )
    
    # Distance and weather interaction
    if 'Distance_km' in df_features.columns and 'Weather_Conditions_Rainy' in df_features.columns:
        df_features['Distance_x_Rainy'] = (
            df_features['Distance_km'] * df_features['Weather_Conditions_Rainy']
        )
    
    # Rush hour and traffic interaction
    if 'Is_Rush_Hour' in df_features.columns and 'Traffic_Conditions_High' in df_features.columns:
        df_features['RushHour_x_HighTraffic'] = (
            df_features['Is_Rush_Hour'] * df_features['Traffic_Conditions_High']
        )
    
    # Order cost and distance interaction
    if 'Order_Cost' in df_features.columns and 'Distance_km' in df_features.columns:
        df_features['OrderCost_per_km'] = df_features['Order_Cost'] / (df_features['Distance_km'] + 0.1)
    
    print("Interaction features created")
    return df_features


def create_aggregated_features(df: pd.DataFrame, group_cols: list, agg_cols: list) -> pd.DataFrame:
    """
    Create aggregated features based on grouping variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_cols (list): Columns to group by
        agg_cols (list): Columns to aggregate
        
    Returns:
        pd.DataFrame: Dataframe with aggregated features
    """
    df_features = df.copy()
    
    for group_col in group_cols:
        if group_col in df_features.columns:
            for agg_col in agg_cols:
                if agg_col in df_features.columns and agg_col != group_col:
                    # Mean aggregation
                    agg_mean = df_features.groupby(group_col)[agg_col].transform('mean')
                    df_features[f'{agg_col}_mean_by_{group_col}'] = agg_mean
                    
                    # Standard deviation aggregation
                    agg_std = df_features.groupby(group_col)[agg_col].transform('std').fillna(0)
                    df_features[f'{agg_col}_std_by_{group_col}'] = agg_std
    
    print(f"Aggregated features created for groups: {group_cols}")
    return df_features


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  method: str = 'standard', feature_cols: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Scale numerical features using specified method.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        method (str): Scaling method ('standard', 'minmax', 'robust')
        feature_cols (list, optional): Specific columns to scale
        
    Returns:
        Tuple[np.ndarray, np.ndarray, scaler]: Scaled training data, scaled test data, fitted scaler
    """
    if feature_cols is None:
        feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Scaling {len(feature_cols)} numerical features using {method} scaling...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    # Fit scaler on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])
    
    print(f"Features scaled successfully. Training set shape: {X_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def create_polynomial_features(df: pd.DataFrame, degree: int = 2, 
                             feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        degree (int): Degree of polynomial features
        feature_cols (list, optional): Columns to create polynomial features for
        
    Returns:
        pd.DataFrame: Dataframe with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    df_poly = df.copy()
    
    if feature_cols is None:
        feature_cols = ['Distance_km', 'Order_Cost']  # Default important features
    
    # Filter existing columns
    feature_cols = [col for col in feature_cols if col in df_poly.columns]
    
    if len(feature_cols) == 0:
        print("No suitable columns found for polynomial features")
        return df_poly
    
    print(f"Creating polynomial features (degree={degree}) for: {feature_cols}")
    
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df_poly[feature_cols])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(feature_cols)
    
    # Add polynomial features to dataframe
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_poly.index)
    
    # Remove original columns to avoid duplication
    for col in feature_cols:
        if col in poly_df.columns:
            poly_df.drop(col, axis=1, inplace=True)
    
    # Concatenate with original dataframe
    df_final = pd.concat([df_poly, poly_df], axis=1)
    
    print(f"Added {len(feature_names) - len(feature_cols)} polynomial features")
    return df_final


def feature_selection_correlation(df: pd.DataFrame, target_col: str, 
                                threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with low correlation to target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        threshold (float): Minimum correlation threshold
        
    Returns:
        pd.DataFrame: Dataframe with selected features
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return df
    
    # Calculate correlations
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Select features above threshold
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    print(f"Selected {len(selected_features)} features with correlation >= {threshold}")
    print(f"Removed {df.shape[1] - len(selected_features)} low-correlation features")
    
    return df[selected_features]


def get_feature_importance_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Get a summary of feature importance based on correlation and basic statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Feature importance summary
    """
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Calculate correlations
    correlations = df[numeric_cols + [target_col]].corr()[target_col].abs()
    
    # Create summary
    summary = pd.DataFrame({
        'Feature': numeric_cols,
        'Correlation_with_Target': [correlations[col] for col in numeric_cols],
        'Mean': [df[col].mean() for col in numeric_cols],
        'Std': [df[col].std() for col in numeric_cols],
        'Missing_Percent': [(df[col].isnull().sum() / len(df)) * 100 for col in numeric_cols]
    })
    
    summary = summary.sort_values('Correlation_with_Target', ascending=False)
    
    print("Feature Importance Summary:")
    print("=" * 50)
    print(summary)
    print("=" * 50)
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Food Delivery Time Prediction")
    
    # Create sample data for testing
    from data_preprocessing import create_sample_dataset
    
    df = create_sample_dataset(100)
    print(f"\\nOriginal dataset shape: {df.shape}")
    
    # Add distance features
    df_with_distance = add_distance_features(df)
    print(f"After distance features: {df_with_distance.shape}")
    
    # Add time features
    df_with_time = add_time_features(df_with_distance)
    print(f"After time features: {df_with_time.shape}")
    
    # Encode categorical variables first
    from data_preprocessing import encode_categorical_variables
    df_encoded = encode_categorical_variables(df_with_time)
    print(f"After encoding: {df_encoded.shape}")
    
    # Create interaction features
    df_final = create_interaction_features(df_encoded)
    print(f"Final feature set: {df_final.shape}")
    
    # Get feature importance summary
    if 'Delivery_Time' in df_final.columns:
        importance_summary = get_feature_importance_summary(df_final, 'Delivery_Time')
    
    print("\\nFeature engineering completed successfully!")