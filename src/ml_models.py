# src/ml_models.py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, classification_report
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class PredictiveModels:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for predictive modeling.
        
        Parameters:
            df (pd.DataFrame): Enhanced customer data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (target)
        """
        # Select features for prediction
        feature_cols = [
            'total_transactions', 'total_spend', 'customer_lifetime_value',
            'avg_transaction_value'
        ] + [col for col in df.columns if 'avg_spend_month' in col]
        
        X = df[feature_cols].values
        y = df['segment'].values
        
        return self.scaler.fit_transform(X), y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train multiple models and compare their performance.
        
        Parameters:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_scores = cross_val_score(self.rf_model, X_train, y_train, cv=5)
        self.rf_model.fit(X_train, y_train)
        
        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(random_state=42)
        gb_scores = cross_val_score(self.gb_model, X_train, y_train, cv=5)
        self.gb_model.fit(X_train, y_train)
        
        # XGBoost
        self.xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_scores = cross_val_score(self.xgb_model, X_train, y_train, cv=5)
        self.xgb_model.fit(X_train, y_train)
        
        # LightGBM
        self.lgb_model = lgb.LGBMClassifier(random_state=42)
        lgb_scores = cross_val_score(self.lgb_model, X_train, y_train, cv=5)
        self.lgb_model.fit(X_train, y_train)
        
        return {
            'random_forest': rf_scores.mean(),
            'gradient_boosting': gb_scores.mean(),
            'xgboost': xgb_scores.mean(),
            'lightgbm': lgb_scores.mean()
        }
    
    def predict_customer_segment(self, 
                               customer_features: np.ndarray, 
                               model_type: str = 'ensemble') -> np.ndarray:
        """
        Predict customer segment using trained models.
        
        Parameters:
            customer_features (np.ndarray): Customer feature matrix
            model_type (str): Type of model to use for prediction
            
        Returns:
            np.ndarray: Predicted segments
        """
        scaled_features = self.scaler.transform(customer_features)
        
        if model_type == 'ensemble':
            # Ensemble prediction using all models
            predictions = np.column_stack([
                self.rf_model.predict(scaled_features),
                self.gb_model.predict(scaled_features),
                self.xgb_model.predict(scaled_features),
                self.lgb_model.predict(scaled_features)
            ])
            return np.round(predictions.mean(axis=1))
        
        # Individual model predictions
        model_map = {
            'random_forest': self.rf_model,
            'gradient_boosting': self.gb_model,
            'xgboost': self.xgb_model,
            'lightgbm': self.lgb_model
        }
        
        return model_map[model_type].predict(scaled_features)
