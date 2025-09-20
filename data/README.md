# Data Directory

This directory contains the datasets used for the Food Delivery Time Prediction project.

## Required Dataset

**File**: `Food_Delivery_Time_Prediction.csv`

### Expected Dataset Structure

The dataset should contain the following columns:

#### Location Data
- `Customer_Lat`: Latitude of customer location (float)
- `Customer_Lng`: Longitude of customer location (float)
- `Restaurant_Lat`: Latitude of restaurant location (float)
- `Restaurant_Lng`: Longitude of restaurant location (float)

#### Environmental Factors
- `Weather_Conditions`: Weather during delivery (categorical: Sunny, Rainy, Cloudy, etc.)
- `Traffic_Conditions`: Traffic level (categorical: Low, Medium, High)

#### Delivery Information
- `Vehicle_Type`: Type of delivery vehicle (categorical: Bike, Scooter, Car, etc.)
- `Delivery_Person_Experience`: Experience level of delivery person (numeric: years)
- `Order_Priority`: Priority level of the order (categorical: Low, Medium, High)

#### Order Details
- `Order_Cost`: Total cost of the order (numeric: currency)
- `Order_Items`: Number of items in the order (numeric)

#### Target Variable
- `Delivery_Time`: Time taken for delivery in minutes (numeric) - **This is our target variable**

### Dataset Requirements

- **Size**: Minimum 1000 records recommended for meaningful analysis
- **Quality**: Data should be clean with minimal missing values
- **Range**: 
  - Delivery times should be between 0-300 minutes (5 hours)
  - Coordinates should be valid latitude/longitude pairs
  - Order costs should be positive values

### Sample Data Generation

If you don't have the actual dataset, the project includes functions to generate sample data:

```python
from src.data_preprocessing import create_sample_dataset
df = create_sample_dataset(n_samples=1000)
df.to_csv('data/Food_Delivery_Time_Prediction.csv', index=False)
```

### Data Privacy

- Ensure all location data is anonymized
- Remove any personally identifiable information (PII)
- Consider data privacy regulations in your region

### Data Sources

Potential sources for this type of data:
- Food delivery companies (Uber Eats, DoorDash, etc.)
- Restaurant delivery services
- Logistics companies
- Public datasets on delivery services
- Simulated/synthetic data based on real-world patterns

## Data Validation

The project includes automated data validation that checks for:
- Required columns presence
- Data types correctness
- Value ranges validity
- Missing values detection
- Duplicate records identification

Run data validation with:
```python
from src.data_preprocessing import get_dataset_info, check_missing_values
df = pd.read_csv('data/Food_Delivery_Time_Prediction.csv')
get_dataset_info(df)
check_missing_values(df)
```

## Processed Data

During the analysis, several processed versions of the data will be created:
- `processed/cleaned_data.csv` - After missing value handling and outlier removal
- `processed/engineered_features.csv` - After feature engineering
- `processed/final_dataset.csv` - Final dataset ready for modeling

These files are automatically generated and should not be committed to version control.