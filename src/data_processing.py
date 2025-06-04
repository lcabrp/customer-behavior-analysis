# src/data_processing.py
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.kmeans = None
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate input data for required columns and data types.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            required_columns (list): List of required column names
            
        Returns:
            bool: True if validation passes
        """
        try:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for minimum data requirements
            if len(df) < 100:
                raise ValueError("Insufficient data for analysis")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            # Log missing value statistics
            missing_stats = df.isnull().sum()
            logger.info(f"Missing values before processing:\n{missing_stats}")
            
            # Handle numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
