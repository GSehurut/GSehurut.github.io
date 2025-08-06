---
layout: post
title: "Machine Learning Applications in Insurance: From Pricing to Claims"
date: 2024-03-23
categories: [Machine Learning, Insurance Analytics]
tags: [Machine Learning, Insurance, Pricing, Claims, Risk Modeling, Python]
image: /pictures/coding.jpg
---

# Machine Learning Applications in Insurance: From Pricing to Claims

## Introduction

Machine learning is revolutionizing the insurance industry, enabling more accurate pricing, better risk assessment, and improved customer experience. This article explores key machine learning applications in insurance, from automated pricing models to claims fraud detection.

## Key Applications

### 1. Automated Pricing Models

Traditional pricing models rely on actuarial tables and manual underwriting. Machine learning enables more sophisticated pricing by incorporating:

- **Telematics Data**: Driving behavior, location, time patterns
- **External Data**: Weather, economic indicators, social media
- **Real-time Factors**: Market conditions, competitor pricing

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class InsurancePricingModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.feature_importance = None
        
    def _initialize_model(self):
        """Initialize the pricing model"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def prepare_features(self, data):
        """Prepare features for pricing model"""
        features = data.copy()
        
        # Create interaction features
        features['age_vehicle_interaction'] = features['age'] * features['vehicle_age']
        features['experience_claims_interaction'] = features['driving_experience'] * features['previous_claims']
        
        # Create categorical encodings
        features = self._encode_categorical(features)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        return features
    
    def _encode_categorical(self, data):
        """Encode categorical variables"""
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            # Use target encoding for high cardinality
            if data[col].nunique() > 10:
                data[f'{col}_encoded'] = self._target_encode(data[col], data['target'])
            else:
                data = pd.get_dummies(data, columns=[col], drop_first=True)
        
        return data
    
    def _target_encode(self, series, target):
        """Target encoding for categorical variables"""
        encoding = target.groupby(series).mean()
        return series.map(encoding)
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # For numerical columns, use median
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, use mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def train(self, X, y):
        """Train the pricing model"""
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(X_processed.columns)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_score': cross_val_score(self.model, X_processed, y, cv=5).mean()
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        X_processed = self.prepare_features(X)
        return self.model.predict(X_processed)
    
    def _calculate_feature_importance(self, feature_names):
        """Calculate feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(feature_names, importance))
        return None

# Example usage
# Load insurance data
data = pd.read_csv('insurance_data.csv')

# Initialize pricing model
pricing_model = InsurancePricingModel(model_type='xgboost')

# Train model
metrics = pricing_model.train(data.drop('premium', axis=1), data['premium'])
print("Model Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Feature importance
if pricing_model.feature_importance:
    importance_df = pd.DataFrame(
        pricing_model.feature_importance.items(),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10))
```

### 2. Claims Fraud Detection

Fraud detection is a critical application of machine learning in insurance:

```python
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionModel:
    def __init__(self, detection_method='isolation_forest'):
        self.detection_method = detection_method
        self.model = self._initialize_detection_model()
        
    def _initialize_detection_model(self):
        """Initialize fraud detection model"""
        if self.detection_method == 'isolation_forest':
            return IsolationForest(
                contamination=0.1,
                random_state=42
            )
        elif self.detection_method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
    
    def extract_fraud_features(self, claims_data):
        """Extract features for fraud detection"""
        features = claims_data.copy()
        
        # Temporal features
        features['claim_time'] = pd.to_datetime(features['claim_date'])
        features['hour_of_day'] = features['claim_time'].dt.hour
        features['day_of_week'] = features['claim_time'].dt.dayofweek
        features['month'] = features['claim_time'].dt.month
        
        # Behavioral features
        features['claims_frequency'] = features.groupby('customer_id')['claim_id'].transform('count')
        features['avg_claim_amount'] = features.groupby('customer_id')['claim_amount'].transform('mean')
        features['claim_amount_std'] = features.groupby('customer_id')['claim_amount'].transform('std')
        
        # Anomaly features
        features['amount_deviation'] = abs(
            features['claim_amount'] - features['avg_claim_amount']
        ) / features['claim_amount_std']
        
        # Location-based features
        features['location_risk'] = features.groupby('location')['claim_amount'].transform('mean')
        
        return features
    
    def detect_fraud(self, claims_data):
        """Detect potential fraud in claims"""
        # Extract features
        features = self.extract_fraud_features(claims_data)
        
        # Select numerical features for modeling
        numerical_features = features.select_dtypes(include=[np.number]).columns
        X = features[numerical_features].fillna(0)
        
        # Fit model and predict
        if self.detection_method == 'isolation_forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            predictions = self.model.fit_predict(X)
            fraud_scores = self.model.decision_function(X)
            
            # Convert to fraud probability
            fraud_probability = 1 - (fraud_scores - fraud_scores.min()) / (fraud_scores.max() - fraud_scores.min())
            
        elif self.detection_method == 'dbscan':
            # DBSCAN returns -1 for noise points (potential fraud)
            predictions = self.model.fit_predict(X)
            fraud_probability = (predictions == -1).astype(float)
        
        return fraud_probability
    
    def analyze_fraud_patterns(self, claims_data, fraud_scores):
        """Analyze patterns in detected fraud"""
        claims_with_scores = claims_data.copy()
        claims_with_scores['fraud_score'] = fraud_scores
        
        # High-risk claims
        high_risk_claims = claims_with_scores[claims_with_scores['fraud_score'] > 0.8]
        
        # Analyze patterns
        patterns = {
            'high_risk_hours': high_risk_claims['hour_of_day'].value_counts().head(),
            'high_risk_locations': high_risk_claims['location'].value_counts().head(),
            'high_risk_amounts': high_risk_claims['claim_amount'].describe(),
            'customer_risk_profile': high_risk_claims.groupby('customer_id')['fraud_score'].mean().sort_values(ascending=False).head()
        }
        
        return patterns

# Example usage
claims_data = pd.read_csv('claims_data.csv')

# Initialize fraud detection model
fraud_model = FraudDetectionModel(detection_method='isolation_forest')

# Detect fraud
fraud_scores = fraud_model.detect_fraud(claims_data)

# Analyze patterns
patterns = fraud_model.analyze_fraud_patterns(claims_data, fraud_scores)

print("Fraud Detection Results:")
print(f"High-risk claims detected: {(fraud_scores > 0.8).sum()}")
print(f"Fraud rate: {(fraud_scores > 0.8).mean():.2%}")

print("\nHigh-risk patterns:")
for pattern, data in patterns.items():
    print(f"\n{pattern}:")
    print(data)
```

### 3. Customer Segmentation and Churn Prediction

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class CustomerAnalytics:
    def __init__(self):
        self.segmentation_model = None
        self.churn_model = None
        self.scaler = StandardScaler()
        
    def segment_customers(self, customer_data, n_clusters=4):
        """Segment customers using clustering"""
        # Prepare features for segmentation
        features = self._prepare_segmentation_features(customer_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        self.segmentation_model = KMeans(n_clusters=n_clusters, random_state=42)
        segments = self.segmentation_model.fit_predict(features_scaled)
        
        # Analyze segments
        segment_analysis = self._analyze_segments(customer_data, segments)
        
        return segments, segment_analysis
    
    def _prepare_segmentation_features(self, data):
        """Prepare features for customer segmentation"""
        features = data[['age', 'income', 'policy_count', 'total_premium', 
                        'claims_count', 'customer_tenure']].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        return features
    
    def _analyze_segments(self, data, segments):
        """Analyze characteristics of each segment"""
        data_with_segments = data.copy()
        data_with_segments['segment'] = segments
        
        segment_analysis = {}
        for segment in range(len(set(segments))):
            segment_data = data_with_segments[data_with_segments['segment'] == segment]
            
            analysis = {
                'size': len(segment_data),
                'avg_age': segment_data['age'].mean(),
                'avg_income': segment_data['income'].mean(),
                'avg_premium': segment_data['total_premium'].mean(),
                'avg_claims': segment_data['claims_count'].mean(),
                'churn_rate': segment_data['churned'].mean() if 'churned' in segment_data.columns else 0
            }
            
            segment_analysis[f'Segment_{segment}'] = analysis
        
        return segment_analysis
    
    def predict_churn(self, customer_data):
        """Predict customer churn"""
        # Prepare features for churn prediction
        features = self._prepare_churn_features(customer_data)
        
        # Split data
        X = features.drop('churned', axis=1)
        y = features['churned']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train churn model
        self.churn_model = LogisticRegression(random_state=42)
        self.churn_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.churn_model.predict(X_test)
        y_prob = self.churn_model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'predictions': y_pred,
            'probabilities': y_prob,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def _prepare_churn_features(self, data):
        """Prepare features for churn prediction"""
        features = data.copy()
        
        # Create churn indicators
        features['days_since_last_claim'] = (pd.Timestamp.now() - pd.to_datetime(features['last_claim_date'])).dt.days
        features['premium_change'] = features['current_premium'] - features['previous_premium']
        features['claims_frequency'] = features['claims_count'] / features['customer_tenure']
        
        # Select relevant features
        feature_columns = ['age', 'income', 'customer_tenure', 'policy_count', 
                          'claims_count', 'days_since_last_claim', 'premium_change', 
                          'claims_frequency', 'churned']
        
        return features[feature_columns].fillna(0)

# Example usage
customer_data = pd.read_csv('customer_data.csv')

# Initialize customer analytics
customer_analytics = CustomerAnalytics()

# Segment customers
segments, segment_analysis = customer_analytics.segment_customers(customer_data)

print("Customer Segmentation Results:")
for segment, analysis in segment_analysis.items():
    print(f"\n{segment}:")
    for metric, value in analysis.items():
        print(f"  {metric}: {value:.2f}")

# Predict churn
churn_results = customer_analytics.predict_churn(customer_data)

print("\nChurn Prediction Results:")
print(churn_results['classification_report'])
```

### 4. Risk Assessment and Underwriting

```python
class RiskAssessmentModel:
    def __init__(self):
        self.risk_models = {}
        
    def build_risk_model(self, line_of_business, historical_data):
        """Build risk assessment model for specific line of business"""
        # Prepare features
        features = self._prepare_risk_features(historical_data)
        
        # Train model
        model = self._train_risk_model(features)
        
        # Store model
        self.risk_models[line_of_business] = model
        
        return model
    
    def _prepare_risk_features(self, data):
        """Prepare features for risk assessment"""
        features = data.copy()
        
        # Create risk indicators
        features['risk_score'] = self._calculate_risk_score(features)
        features['exposure_factor'] = features['sum_insured'] / features['premium']
        features['loss_ratio'] = features['total_claims'] / features['total_premium']
        
        return features
    
    def _calculate_risk_score(self, data):
        """Calculate composite risk score"""
        # This is a simplified risk scoring approach
        risk_factors = {
            'age': lambda x: np.where(x < 25, 1.5, np.where(x > 65, 1.3, 1.0)),
            'claims_history': lambda x: np.where(x > 3, 1.4, 1.0),
            'location_risk': lambda x: np.where(x > 0.8, 1.2, 1.0)
        }
        
        risk_score = 1.0
        for factor, calculation in risk_factors.items():
            if factor in data.columns:
                risk_score *= calculation(data[factor])
        
        return risk_score
    
    def _train_risk_model(self, features):
        """Train risk assessment model"""
        # Use gradient boosting for risk modeling
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        X = features.drop(['risk_score', 'target'], axis=1, errors='ignore')
        y = features['risk_score']
        
        model.fit(X, y)
        return model
    
    def assess_risk(self, line_of_business, application_data):
        """Assess risk for new application"""
        if line_of_business not in self.risk_models:
            raise ValueError(f"No model trained for {line_of_business}")
        
        model = self.risk_models[line_of_business]
        
        # Prepare features
        features = self._prepare_risk_features(application_data)
        X = features.drop(['risk_score', 'target'], axis=1, errors='ignore')
        
        # Predict risk
        risk_score = model.predict(X)
        
        return risk_score

# Example usage
risk_model = RiskAssessmentModel()

# Build risk model for auto insurance
auto_data = pd.read_csv('auto_insurance_data.csv')
auto_risk_model = risk_model.build_risk_model('auto', auto_data)

# Assess risk for new application
new_application = pd.DataFrame({
    'age': [30],
    'driving_experience': [5],
    'vehicle_type': ['sedan'],
    'location': ['urban'],
    'sum_insured': [25000]
})

risk_score = risk_model.assess_risk('auto', new_application)
print(f"Risk Score: {risk_score[0]:.2f}")
```

## Model Deployment and Monitoring

### 1. Model Deployment

```python
import joblib
import json
from datetime import datetime

class ModelDeployment:
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        
    def deploy_model(self, model, model_name, version, metadata=None):
        """Deploy model to production"""
        # Create model directory
        model_dir = f"{self.model_path}{model_name}/v{version}/"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = f"{model_dir}model.pkl"
        joblib.dump(model, model_file)
        
        # Save metadata
        metadata_file = f"{model_dir}metadata.json"
        metadata = metadata or {}
        metadata.update({
            'version': version,
            'deployment_date': datetime.now().isoformat(),
            'model_type': type(model).__name__
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_dir
    
    def load_model(self, model_name, version):
        """Load deployed model"""
        model_dir = f"{self.model_path}{model_name}/v{version}/"
        model_file = f"{model_dir}model.pkl"
        
        if os.path.exists(model_file):
            return joblib.load(model_file)
        else:
            raise FileNotFoundError(f"Model not found: {model_file}")
```

### 2. Model Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.metrics_history = []
        
    def monitor_model_performance(self, model, X_test, y_test, model_name):
        """Monitor model performance"""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Check for performance degradation
        self._check_performance_degradation(metrics)
        
        return metrics
    
    def _check_performance_degradation(self, current_metrics):
        """Check for performance degradation"""
        if len(self.metrics_history) > 1:
            previous_metrics = self.metrics_history[-2]
            
            # Calculate performance change
            performance_change = {
                'accuracy_change': current_metrics['accuracy'] - previous_metrics['accuracy'],
                'precision_change': current_metrics['precision'] - previous_metrics['precision'],
                'recall_change': current_metrics['recall'] - previous_metrics['recall']
            }
            
            # Alert if significant degradation
            threshold = -0.05
            for metric, change in performance_change.items():
                if change < threshold:
                    self._send_alert(f"Performance degradation detected: {metric} decreased by {abs(change):.3f}")
    
    def _send_alert(self, message):
        """Send alert for model issues"""
        print(f"ALERT: {message}")
        # Implement actual alerting (email, Slack, etc.)
```

## Conclusion

Machine learning is transforming the insurance industry by enabling more accurate pricing, better fraud detection, and improved customer experience. The key to success lies in:

1. **Data Quality**: Ensuring high-quality, relevant data
2. **Model Interpretability**: Making models explainable for regulatory compliance
3. **Continuous Monitoring**: Tracking model performance and drift
4. **Ethical Considerations**: Ensuring fair and unbiased models

### Future Trends

1. **Explainable AI**: Making ML models more interpretable
2. **Federated Learning**: Collaborative model training across organizations
3. **Real-time Processing**: Stream processing for instant decisions
4. **AI Ethics**: Ensuring fair and responsible AI in insurance

Would you like to explore any specific aspect of machine learning in insurance in more detail? Feel free to reach out with questions or suggestions for future topics. 