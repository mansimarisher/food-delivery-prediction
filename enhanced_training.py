"""
Enhanced Training Script for Food Delivery Time Prediction

This script implements advanced machine learning models and techniques to achieve
accuracy > 70% and R² > 0.60 for food delivery time prediction.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.cluster import KMeans

# XGBoost and LightGBM (install if not available)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Local imports
from src.data_preprocessing import load_dataset, handle_missing_values, encode_categorical_variables
from src.feature_engineering import add_distance_features, add_time_features

warnings.filterwarnings('ignore')

class EnhancedFoodDeliveryPredictor:
    """Enhanced predictor with multiple advanced models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_regression = {}
        self.models_classification = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.results = {}
        
    def parse_location(self, location_str: str) -> Tuple[float, float]:
        """Parse location string to lat, lng coordinates."""
        try:
            # Remove parentheses and split
            clean_str = location_str.strip("()")
            lat, lng = map(float, clean_str.split(", "))
            return lat, lng
        except:
            return np.nan, np.nan
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better model performance."""
        df_enhanced = df.copy()
        
        print("Creating advanced features...")
        
        # Parse location coordinates
        df_enhanced[['Customer_Lat', 'Customer_Lng']] = df_enhanced['Customer_Location'].apply(
            lambda x: pd.Series(self.parse_location(x))
        )
        df_enhanced[['Restaurant_Lat', 'Restaurant_Lng']] = df_enhanced['Restaurant_Location'].apply(
            lambda x: pd.Series(self.parse_location(x))
        )
        
        # Distance features
        df_enhanced = add_distance_features(df_enhanced)
        
        # Time-based features from Order_Time
        time_mapping = {
            'Morning': 8, 'Afternoon': 14, 'Evening': 19, 'Night': 22
        }
        df_enhanced['Hour'] = df_enhanced['Order_Time'].map(time_mapping)
        df_enhanced['Is_Rush_Hour'] = df_enhanced['Hour'].apply(
            lambda x: 1 if x in [12, 13, 19, 20] else 0
        )
        
        # Weather-Traffic interactions
        weather_severity = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3, 'Snowy': 4}
        traffic_severity = {'Low': 1, 'Medium': 2, 'High': 3}
        
        df_enhanced['Weather_Severity'] = df_enhanced['Weather_Conditions'].map(weather_severity)
        df_enhanced['Traffic_Severity'] = df_enhanced['Traffic_Conditions'].map(traffic_severity)
        df_enhanced['Weather_Traffic_Product'] = df_enhanced['Weather_Severity'] * df_enhanced['Traffic_Severity']
        
        # Vehicle efficiency (based on traffic and weather)
        vehicle_efficiency = {'Bike': 3, 'Bicycle': 2, 'Car': 1}  # Higher = more affected by conditions
        df_enhanced['Vehicle_Efficiency'] = df_enhanced['Vehicle_Type'].map(vehicle_efficiency)
        df_enhanced['Condition_Impact'] = df_enhanced['Vehicle_Efficiency'] * df_enhanced['Weather_Traffic_Product']
        
        # Priority encoding
        priority_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df_enhanced['Priority_Numeric'] = df_enhanced['Order_Priority'].map(priority_map)
        
        # Cost per km
        df_enhanced['Cost_per_km'] = df_enhanced['Order_Cost'] / (df_enhanced['Distance_km'] + 0.1)
        
        # Rating interactions
        df_enhanced['Rating_Product'] = df_enhanced['Restaurant_Rating'] * df_enhanced['Customer_Rating']
        df_enhanced['Rating_Difference'] = abs(df_enhanced['Restaurant_Rating'] - df_enhanced['Customer_Rating'])
        
        # Experience-distance interaction
        df_enhanced['Experience_Distance_Ratio'] = df_enhanced['Delivery_Person_Experience'] / (df_enhanced['Distance_km'] + 1)
        
        # Location clustering for customer and restaurant zones
        if 'Customer_Lat' in df_enhanced.columns and not df_enhanced['Customer_Lat'].isna().all():
            customer_coords = df_enhanced[['Customer_Lat', 'Customer_Lng']].dropna()
            if len(customer_coords) > 10:
                kmeans_customer = KMeans(n_clusters=5, random_state=self.random_state)
                df_enhanced.loc[customer_coords.index, 'Customer_Zone'] = kmeans_customer.fit_predict(customer_coords)
                
            restaurant_coords = df_enhanced[['Restaurant_Lat', 'Restaurant_Lng']].dropna()
            if len(restaurant_coords) > 10:
                kmeans_restaurant = KMeans(n_clusters=5, random_state=self.random_state)
                df_enhanced.loc[restaurant_coords.index, 'Restaurant_Zone'] = kmeans_restaurant.fit_predict(restaurant_coords)
        
        # Polynomial features for key variables
        poly_features = ['Distance_km', 'Order_Cost', 'Delivery_Person_Experience']
        for feature in poly_features:
            if feature in df_enhanced.columns:
                df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
                df_enhanced[f'{feature}_log'] = np.log1p(df_enhanced[feature])
        
        print(f"Enhanced features created. New shape: {df_enhanced.shape}")
        return df_enhanced
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Delivery_Time') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling with advanced preprocessing."""
        
        # Handle missing values
        df_clean = handle_missing_values(df, strategy='median')
        
        # Advanced feature engineering
        df_features = self.advanced_feature_engineering(df_clean)
        
        # Handle categorical features - convert Distance_Category to numeric first
        if 'Distance_Category' in df_features.columns:
            df_features['Distance_Category_Numeric'] = df_features['Distance_Category'].astype(str).map({
                'Very Close': 1, 'Close': 2, 'Medium': 3, 'Far': 4
            }).fillna(2)  # Default to 'Close'
            df_features = df_features.drop('Distance_Category', axis=1)
        
        # Encode categorical variables
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in [target_col, 'Order_ID', 'Customer_Location', 'Restaurant_Location']]
        
        # Convert categorical columns to string first
        for col in categorical_cols:
            df_features[col] = df_features[col].astype(str)
        
        if categorical_cols:
            df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
        else:
            df_encoded = df_features
        
        # Remove non-feature columns
        cols_to_remove = ['Order_ID', 'Customer_Location', 'Restaurant_Location']
        df_encoded = df_encoded.drop([col for col in cols_to_remove if col in df_encoded.columns], axis=1)
        
        # Ensure all columns are numeric
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                print(f"Warning: Column {col} is still non-numeric. Converting to numeric.")
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        
        # Fill any remaining NaN values
        df_encoded = df_encoded.fillna(df_encoded.median())
        
        # Separate features and target
        if target_col in df_encoded.columns:
            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]
        else:
            X = df_encoded
            y = None
        
        return X, y
    
    def create_regression_models(self) -> Dict[str, Any]:
        """Create dictionary of regression models."""
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state)
        }
        
        if HAS_XGB:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
        
        if HAS_LGB:
            models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        
        return models
    
    def create_classification_models(self) -> Dict[str, Any]:
        """Create dictionary of classification models."""
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svc': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state)
        }
        
        if HAS_XGB:
            models['xgboost'] = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='logloss')
        
        if HAS_LGB:
            models['lightgbm'] = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
        
        return models
    
    def optimize_hyperparameters(self, model, param_grid: Dict, X_train, y_train, cv: int = 5) -> Any:
        """Optimize hyperparameters using GridSearchCV."""
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='neg_mean_squared_error' if hasattr(model, 'predict') else 'accuracy',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def train_regression_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict]:
        """Train and evaluate regression models."""
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['regression'] = scaler
        
        models = self.create_regression_models()
        results = {}
        
        print("\n=== TRAINING REGRESSION MODELS ===")
        
        # Hyperparameter grids for key models
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        }
        
        if HAS_XGB:
            param_grids['xgboost'] = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for models that benefit from it
                if name in ['linear_regression', 'svr', 'mlp']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Hyperparameter optimization for selected models
                if name in param_grids:
                    model = self.optimize_hyperparameters(model, param_grids[name], X_train_model, y_train)
                
                # Train model
                model.fit(X_train_model, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'y_pred': y_pred
                }
                
                print(f"{name:15} - R²: {r2:.4f}, MAE: {mae:.2f}, CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Create ensemble models
        if len(results) >= 3:
            print("\nTraining ensemble models...")
            
            # Voting regressor
            voting_models = [(name, result['model']) for name, result in results.items() if name != 'mlp'][:3]
            if len(voting_models) >= 2:
                voting_reg = VotingRegressor(voting_models)
                voting_reg.fit(X_train, y_train)
                y_pred_voting = voting_reg.predict(X_test)
                
                results['voting'] = {
                    'model': voting_reg,
                    'mse': mean_squared_error(y_test, y_pred_voting),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_voting)),
                    'mae': mean_absolute_error(y_test, y_pred_voting),
                    'r2': r2_score(y_test, y_pred_voting),
                    'cv_r2_mean': 0,  # Skip CV for ensemble
                    'cv_r2_std': 0,
                    'y_pred': y_pred_voting
                }
                
                print(f"{'Voting':15} - R²: {results['voting']['r2']:.4f}, MAE: {results['voting']['mae']:.2f}")
        
        self.models_regression = {name: result['model'] for name, result in results.items()}
        return results
    
    def train_classification_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict]:
        """Train and evaluate classification models."""
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['classification'] = scaler
        
        models = self.create_classification_models()
        results = {}
        
        print("\n=== TRAINING CLASSIFICATION MODELS ===")
        
        # Hyperparameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        }
        
        if HAS_XGB:
            param_grids['xgboost'] = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for models that benefit from it
                if name in ['logistic_regression', 'svc', 'mlp']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Hyperparameter optimization for selected models
                if name in param_grids:
                    model = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
                    model.fit(X_train_model, y_train)
                    model = model.best_estimator_
                else:
                    model.fit(X_train_model, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'cv_accuracy_mean': cv_mean,
                    'cv_accuracy_std': cv_std,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"{name:15} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV Acc: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.models_classification = {name: result['model'] for name, result in results.items()}
        return results
    
    def plot_results(self, regression_results: Dict, classification_results: Dict, y_test_reg=None):
        """Create comprehensive plots of model performance."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Food Delivery Prediction Model Results', fontsize=16, y=0.95)
        
        # Regression R² scores
        reg_names = list(regression_results.keys())
        reg_r2_scores = [regression_results[name]['r2'] for name in reg_names]
        
        axes[0, 0].barh(reg_names, reg_r2_scores, color='skyblue')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Regression Model Performance (R²)')
        axes[0, 0].axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Target: 0.6')
        axes[0, 0].legend()
        
        # Regression MAE
        reg_mae_scores = [regression_results[name]['mae'] for name in reg_names]
        axes[0, 1].barh(reg_names, reg_mae_scores, color='lightcoral')
        axes[0, 1].set_xlabel('Mean Absolute Error (minutes)')
        axes[0, 1].set_title('Regression Model Performance (MAE)')
        
        # Classification Accuracy
        clf_names = list(classification_results.keys())
        clf_acc_scores = [classification_results[name]['accuracy'] for name in clf_names]
        
        axes[0, 2].barh(clf_names, clf_acc_scores, color='lightgreen')
        axes[0, 2].set_xlabel('Accuracy')
        axes[0, 2].set_title('Classification Model Performance (Accuracy)')
        axes[0, 2].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Target: 0.7')
        axes[0, 2].legend()
        
        # Best regression model predictions vs actual
        if y_test_reg is not None:
            best_reg_model = max(regression_results.keys(), key=lambda k: regression_results[k]['r2'])
            best_reg_pred = regression_results[best_reg_model]['y_pred']
            
            axes[1, 0].scatter(y_test_reg, best_reg_pred, alpha=0.6)
            axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
            axes[1, 0].set_xlabel('Actual Delivery Time')
            axes[1, 0].set_ylabel('Predicted Delivery Time')
            axes[1, 0].set_title(f'Best Regression Model: {best_reg_model} (R²: {regression_results[best_reg_model]["r2"]:.3f})')
        else:
            axes[1, 0].text(0.5, 0.5, 'No test data available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Predictions vs Actual')
        
        # Model comparison - R² vs MAE
        axes[1, 1].scatter(reg_r2_scores, reg_mae_scores, s=100)
        for i, name in enumerate(reg_names):
            axes[1, 1].annotate(name, (reg_r2_scores[i], reg_mae_scores[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].set_ylabel('MAE (minutes)')
        axes[1, 1].set_title('Regression Models: R² vs MAE')
        
        # Classification F1 scores
        clf_f1_scores = [classification_results[name]['f1'] for name in clf_names]
        axes[1, 2].barh(clf_names, clf_f1_scores, color='orange')
        axes[1, 2].set_xlabel('F1 Score')
        axes[1, 2].set_title('Classification Model Performance (F1)')
        
        plt.tight_layout()
        plt.savefig('enhanced_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, regression_results: Dict, classification_results: Dict):
        """Save the best performing models."""
        
        os.makedirs('models', exist_ok=True)
        
        # Save best regression model
        if regression_results:
            best_reg_model = max(regression_results.keys(), key=lambda k: regression_results[k]['r2'])
            best_reg_r2 = regression_results[best_reg_model]['r2']
            
            joblib.dump(regression_results[best_reg_model]['model'], f'models/best_regression_model_{best_reg_model}.pkl')
            joblib.dump(self.scalers.get('regression'), f'models/regression_scaler.pkl')
            
            print(f"Best regression model saved: {best_reg_model} (R²: {best_reg_r2:.4f})")
        
        # Save best classification model
        if classification_results:
            best_clf_model = max(classification_results.keys(), key=lambda k: classification_results[k]['accuracy'])
            best_clf_acc = classification_results[best_clf_model]['accuracy']
            
            joblib.dump(classification_results[best_clf_model]['model'], f'models/best_classification_model_{best_clf_model}.pkl')
            joblib.dump(self.scalers.get('classification'), f'models/classification_scaler.pkl')
            
            print(f"Best classification model saved: {best_clf_model} (Accuracy: {best_clf_acc:.4f})")
    
    def generate_report(self, regression_results: Dict, classification_results: Dict):
        """Generate comprehensive performance report."""
        
        report = []
        report.append("="*80)
        report.append("ENHANCED FOOD DELIVERY PREDICTION MODEL REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Regression results
        report.append("REGRESSION RESULTS (Delivery Time Prediction)")
        report.append("-" * 50)
        for name, results in regression_results.items():
            report.append(f"{name:20}: R²={results['r2']:.4f}, MAE={results['mae']:.2f}, RMSE={results['rmse']:.2f}")
        
        # Best regression model
        if regression_results:
            best_reg = max(regression_results.keys(), key=lambda k: regression_results[k]['r2'])
            best_r2 = regression_results[best_reg]['r2']
            report.append(f"\nBest Regression Model: {best_reg} (R²: {best_r2:.4f})")
            report.append(f"Target R² >= 0.60: {'ACHIEVED' if best_r2 >= 0.60 else 'NOT ACHIEVED'}")
        
        report.append("")
        
        # Classification results
        report.append("CLASSIFICATION RESULTS (Fast/Slow Delivery)")
        report.append("-" * 50)
        for name, results in classification_results.items():
            report.append(f"{name:20}: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}, AUC={results['auc']:.4f}")
        
        # Best classification model
        if classification_results:
            best_clf = max(classification_results.keys(), key=lambda k: classification_results[k]['accuracy'])
            best_acc = classification_results[best_clf]['accuracy']
            report.append(f"\nBest Classification Model: {best_clf} (Accuracy: {best_acc:.4f})")
            report.append(f"Target Accuracy >= 0.70: {'ACHIEVED' if best_acc >= 0.70 else 'NOT ACHIEVED'}")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        with open('enhanced_model_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))

def main():
    """Main execution function."""
    print("Enhanced Food Delivery Prediction Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EnhancedFoodDeliveryPredictor(random_state=42)
    
    # Load and prepare data
    print("Loading dataset...")
    df = load_dataset('data/Food_Delivery_Time_Prediction.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target statistics:\n{df['Delivery_Time'].describe()}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = predictor.prepare_data(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Features: {list(X.columns[:10])}...")  # Show first 10 features
    
    # Split data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train regression models
    regression_results = predictor.train_regression_models(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    # Prepare classification data (fast vs slow delivery)
    threshold = y.median()
    y_class = (y > threshold).astype(int)
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Train classification models
    classification_results = predictor.train_classification_models(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    # Generate visualizations
    predictor.plot_results(regression_results, classification_results, y_test_reg)
    
    # Save models
    predictor.save_models(regression_results, classification_results)
    
    # Generate report
    predictor.generate_report(regression_results, classification_results)
    
    print("\nEnhanced training completed!")
    print("Check 'enhanced_model_results.png' for visualizations")
    print("Check 'enhanced_model_report.txt' for detailed results")

if __name__ == "__main__":
    main()