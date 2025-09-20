# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Food Delivery Time Prediction** machine learning project that uses linear and logistic regression to predict delivery times and classify deliveries as fast/delayed. The project analyzes factors like customer location, restaurant location, weather conditions, traffic patterns, and other variables to optimize delivery operations.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Alternative: Create virtual environment first
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Analysis

**Primary Jupyter Notebook:**
```bash
# Start Jupyter and run main analysis
jupyter notebook notebooks/Food_Delivery_Time_Prediction.ipynb
```

**Python Modules (Alternative execution):**
```bash
# Run individual modules
python src/data_preprocessing.py
python src/feature_engineering.py 
python src/model_training.py

# Run from project root with module import
python -m src.data_preprocessing
python -m src.model_training
```

### Testing and Development
```bash
# Run tests (if test files exist)
pytest tests/

# Code quality checks
flake8 src/
black src/ --check

# Apply code formatting
black src/
```

### Configuration Management
```bash
# Generate and save current config
python config/model_config.py

# The config system is based on dataclasses and supports JSON serialization
```

## Architecture Overview

### Core Module Structure

**`src/data_preprocessing.py`** - Data loading, cleaning, and preprocessing pipeline
- Handles missing values with configurable strategies (median/mean/mode)
- Outlier detection using IQR or Z-score methods
- Categorical encoding (one-hot or label encoding)
- Can generate synthetic sample data if original dataset missing

**`src/feature_engineering.py`** - Advanced feature creation and transformation
- Haversine distance calculations between coordinates using `geopy`
- Time-based features (rush hour, meal time, weekend flags)
- Interaction features (distance × traffic, weather × distance)
- Feature scaling with multiple methods (standard/minmax/robust)
- Polynomial feature generation capability

**`src/model_training.py`** - Model training, evaluation, and hyperparameter tuning
- Linear regression for continuous delivery time prediction
- Logistic regression for binary fast/delayed classification
- Cross-validation and hyperparameter tuning with GridSearchCV
- Model persistence using joblib
- Comprehensive evaluation metrics and visualization

**`src/utils.py`** - Visualization, reporting, and utility functions
- Plotting utilities with consistent styling using matplotlib/seaborn
- Business metrics calculation (cost impact, accuracy thresholds)
- Model performance reporting and executive summary generation
- Experiment result saving/loading in JSON format

**`config/model_config.py`** - Centralized configuration management
- Dataclass-based configuration system with sections for data, features, models, visualization, and experiments
- Hyperparameter grids for model tuning
- Data validation rules and performance benchmarks
- JSON serialization support for configuration persistence

### Data Flow Architecture

1. **Data Loading** (`data_preprocessing.py`) → Dataset or synthetic data generation
2. **Feature Engineering** (`feature_engineering.py`) → Distance, time, and interaction features
3. **Model Training** (`model_training.py`) → Regression and classification models
4. **Evaluation & Reporting** (`utils.py`) → Metrics, visualizations, business impact

### Key Design Patterns

- **Configuration-driven development**: Extensive use of `model_config.py` for parameterization
- **Modular pipeline**: Each module can run independently or be imported by others
- **Dual modeling approach**: Both regression (continuous) and classification (binary) models
- **Business-focused metrics**: Beyond accuracy, includes cost impact and time-based accuracy thresholds
- **Reproducible research**: Consistent random seeds and configuration persistence

## Important Dataset Information

**Expected CSV Structure:**
- `Customer_Lat`, `Customer_Lng` - Customer coordinates
- `Restaurant_Lat`, `Restaurant_Lng` - Restaurant coordinates  
- `Weather_Conditions` - Weather categories (Sunny/Rainy/Cloudy)
- `Traffic_Conditions` - Traffic levels (Low/Medium/High)
- `Vehicle_Type` - Delivery vehicle (Bike/Scooter/Car)
- `Delivery_Person_Experience` - Experience level (1-10)
- `Order_Cost` - Order value
- `Order_Priority` - Priority level (Low/Medium/High)
- `Delivery_Time` - Target variable (minutes)

**Data Fallback**: If `data/Food_Delivery_Time_Prediction.csv` is missing, the system automatically generates synthetic sample data for development and testing.

## Configuration System

The project uses a sophisticated configuration system in `config/model_config.py`:

- **DataConfig**: Paths, test split, missing value strategies, outlier handling
- **FeatureConfig**: Distance calculations, time features, scaling, polynomial features
- **ModelConfig**: Target column, normalization, hyperparameter tuning settings
- **VisualizationConfig**: Plot styling, figure sizes, save formats
- **ExperimentConfig**: Project metadata, logging, reproducibility settings

Access via: `from config.model_config import default_config`

## Model Training Workflow

1. **Data Preparation**: Load/create data → handle missing values → detect outliers
2. **Feature Engineering**: Distance calculation → time features → categorical encoding → interactions
3. **Model Training**: Train-test split → linear regression → logistic regression → evaluation
4. **Results**: Performance metrics → visualizations → business impact analysis → model persistence

The system automatically handles:
- Feature scaling and normalization
- Cross-validation for robust evaluation  
- Hyperparameter tuning when enabled
- Model serialization for deployment
- Comprehensive reporting with business metrics

## Development Notes

- All modules include comprehensive docstrings and type hints
- The system is designed to work with or without the actual dataset
- Models are automatically saved to `models/` directory when training completes
- All visualizations can be automatically saved to `reports/` directory
- The configuration system allows easy experimentation with different parameters
- Business metrics focus on delivery time accuracy within practical thresholds (5-10 minutes)