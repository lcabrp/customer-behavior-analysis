# tests/test_data_processing.py
import unittest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'amount': np.random.normal(100, 20, 100),
            'transaction_date': pd.date_range(start='2023-01-01', periods=100)
        })
    
    def test_validate_data(self):
        required_cols = ['customer_id', 'amount', 'transaction_date']
        self.assertTrue(self.processor.validate_data(self.sample_data, required_cols))
        
        # Test missing columns
        invalid_data = self.sample_data.drop('amount', axis=1)
        with self.assertRaises(ValueError):
            self.processor.validate_data(invalid_data, required_cols)
    
    def test_handle_missing_values(self):
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:10, 'amount'] = np.nan
        
        processed_data = self.processor.handle_missing_values(data_with_missing)
        self.assertEqual(processed_data.isnull().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()
