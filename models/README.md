# Models Directory

This directory contains trained machine learning models and their associated files for the Food Delivery Time Prediction project.

## Model Files

The following files will be generated during model training:

### Linear Regression Model (Delivery Time Prediction)
- `linear_regression_model.pkl` - Trained linear regression model
- `feature_scaler.pkl` - StandardScaler used for feature normalization

### Logistic Regression Model (Fast/Delayed Classification)  
- `logistic_regression_model.pkl` - Trained logistic regression model
- `classification_scaler.pkl` - StandardScaler used for feature normalization

### Model Metadata
- `model_results.json` - Model performance metrics and metadata
- `feature_importance.json` - Feature importance rankings
- `training_log.txt` - Training process logs

## Model Performance

### Expected Performance Benchmarks

**Linear Regression (Delivery Time Prediction):**
- R² Score: > 0.6 (good), > 0.8 (excellent)
- Mean Absolute Error: < 10 minutes (acceptable)
- Root Mean Squared Error: < 15 minutes

**Logistic Regression (Fast/Delayed Classification):**
- Accuracy: > 0.75 (good), > 0.85 (excellent)
- F1-Score: > 0.75
- AUC-ROC: > 0.80

## Loading Trained Models

### Using Built-in Functions

```python
from src.model_training import load_model_and_scaler

# Load linear regression model
lr_model, lr_scaler = load_model_and_scaler('linear_regression')

# Load logistic regression model  
logistic_model, logistic_scaler = load_model_and_scaler('logistic_regression')
```

### Manual Loading

```python
import joblib

# Load models
lr_model = joblib.load('models/linear_regression_model.pkl')
logistic_model = joblib.load('models/logistic_regression_model.pkl')

# Load scalers
scaler = joblib.load('models/feature_scaler.pkl')
```

## Making Predictions

### Delivery Time Prediction

```python
import numpy as np

# Prepare your features (same format as training data)
new_data = [...]  # Your feature vector

# Scale features if scaler was used
if lr_scaler:
    new_data_scaled = lr_scaler.transform([new_data])
else:
    new_data_scaled = [new_data]

# Predict delivery time
predicted_time = lr_model.predict(new_data_scaled)[0]
print(f"Predicted delivery time: {predicted_time:.1f} minutes")
```

### Fast/Delayed Classification

```python
# Predict probability of delay
delay_probability = logistic_model.predict_proba(new_data_scaled)[0][1]

# Predict class (0 = Fast, 1 = Delayed)
delay_class = logistic_model.predict(new_data_scaled)[0]

print(f"Delay probability: {delay_probability:.2f}")
print(f"Classification: {'Delayed' if delay_class == 1 else 'Fast'}")
```

## Model Versioning

Models are saved with timestamps and version information:
- Include training date in model names
- Track model performance over time
- Keep previous versions for comparison

## Model Deployment Considerations

### Production Readiness Checklist
- [ ] Model validation on hold-out test set
- [ ] Feature importance analysis completed
- [ ] Model interpretability documented
- [ ] Performance benchmarks met
- [ ] Error analysis conducted
- [ ] Model monitoring strategy defined

### API Integration
For production deployment, consider:
- Model serving frameworks (FastAPI, Flask)
- Input validation and preprocessing
- Error handling and logging
- Model versioning and rollback capability
- Performance monitoring

### Model Updates
- Retrain models periodically with new data
- Monitor for model drift
- A/B test new model versions
- Maintain model performance logs

## File Formats

All models are saved in pickle format (.pkl) for compatibility with scikit-learn. Alternative formats:
- Joblib (more efficient for numpy arrays)
- ONNX (for cross-platform deployment)
- JSON (for simple models and metadata)

## Security Notes

- Models may contain sensitive information about training data
- Ensure secure storage and access controls
- Consider model encryption for sensitive applications
- Regular security audits of model files