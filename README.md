# Food Delivery Time Prediction Project

A comprehensive machine learning project to predict food delivery times using linear regression and logistic regression models. This project analyzes factors such as customer location, restaurant location, weather conditions, traffic patterns, and other variables to optimize delivery operations.

## Project Objective

The goal is to predict food delivery times based on customer location, restaurant location, weather, traffic, and other factors. This involves both data preprocessing and building predictive models using linear regression and logistic regression.

## Project Structure

```
food-delivery-prediction/
├── data/                           # Dataset storage
│   └── Food_Delivery_Time_Prediction.csv
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation and transformation
│   ├── model_training.py         # Model training functions
│   ├── model_evaluation.py       # Evaluation metrics and validation
│   └── utils.py                   # Utility functions
├── notebooks/                     # Jupyter notebooks
│   └── Food_Delivery_Time_Prediction.ipynb
├── models/                        # Trained model artifacts
├── config/                        # Configuration files
│   └── model_config.py
├── tests/                         # Unit tests
├── reports/                       # Generated reports and visualizations
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

## Phase 1: Data Collection and Exploratory Data Analysis (EDA)

### Step 1 - Data Import and Preprocessing
- **Dataset**: Load `Food_Delivery_Time_Prediction.csv`
- **Handle Missing Values**: Check and handle missing values in Distance, Delivery_Time, etc.
- **Data Transformation**:
  - Encode categorical variables (Weather Conditions, Traffic Conditions, Vehicle Type)
  - Normalize/standardize numeric columns (Distance, Delivery_Time, Order_Cost)

### Step 2 - Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Calculate mean, median, mode, variance for numerical features
- **Correlation Analysis**: Visualize correlations with target variable (Delivery_Time)
- **Outlier Detection**: Detect and handle outliers using boxplots

### Step 3 - Feature Engineering
- **Distance Calculation**: Calculate distance using Haversine formula if needed
- **Time-Based Features**: Create Rush Hour vs Non-Rush Hour features

## Phase 2: Predictive Modeling

### Step 4 - Linear Regression Model
- **Train-Test Split**: 80/20 split
- **Model Building**: Predict Delivery Time based on Distance, Traffic_Conditions, Order_Priority
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R²)
  - Mean Absolute Error (MAE)

### Step 5 - Logistic Regression Model
- **Model Objective**: Classify deliveries as "Fast" or "Delayed"
- **Model Implementation**: Binary classification using traffic, weather, experience features
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix

## Phase 3: Reporting and Insights

### Step 6 - Model Evaluation and Comparison
- Compare Linear Regression and Logistic Regression performance
- Visualize results using confusion matrices and ROC curves

### Step 7 - Actionable Insights
Based on model predictions, suggest:
- Optimizing delivery routes
- Adjusting staffing during high-traffic periods
- Providing better training to delivery staff

## Final Deliverables

1. **Jupyter Notebook (.ipynb)**: Complete code for data preprocessing, model training, and evaluation
2. **Data Visualizations**: Scatter plots, pair plots, confusion matrices, ROC curves
3. **Final Report**: Detailed summary with dataset description, preprocessing steps, model evaluation, and actionable insights

## Getting Started

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Place Dataset**:
   - Add `Food_Delivery_Time_Prediction.csv` to the `data/` directory

3. **Run the Analysis**:
```bash
jupyter notebook notebooks/Food_Delivery_Time_Prediction.ipynb
```

4. **Alternative: Run Python Scripts**:
```bash
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
```

## Key Features

- **Data Preprocessing**: Automated handling of missing values and categorical encoding
- **Feature Engineering**: Distance calculations and time-based feature creation
- **Multiple Models**: Linear regression for continuous prediction, logistic regression for classification
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Actionable Insights**: Business recommendations based on model results

## Expected Dataset Columns

- Customer and restaurant coordinates (lat/lng)
- Weather conditions
- Traffic conditions  
- Vehicle type
- Delivery person experience
- Order details (cost, priority)
- Target: Delivery time

## Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- jupyter
- scipy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request
