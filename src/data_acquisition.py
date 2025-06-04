# src/data_acquisition.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
from typing import Tuple
import logging

class DataAcquisition:
    def __init__(self, data_dir: str = './data'):
        """
        Initialize data acquisition paths.
        
        Parameters:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_kaggle_dataset(self) -> str:
        """
        Download retail dataset from Kaggle.
        Note: Requires kaggle API credentials in ~/.kaggle/kaggle.json
        
        Returns:
            str: Path to downloaded file
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download retail dataset (replace with actual dataset details)
            api.dataset_download_files(
                'dataset/retail-data-transactions',
                path=self.raw_dir,
                unzip=True
            )
            
            return os.path.join(self.raw_dir, 'transactions.csv')
            
        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset: {str(e)}")
            # If download fails, we'll generate synthetic data instead
            return self.generate_synthetic_data()

    def generate_synthetic_data(self) -> Tuple[str, str]:
        """
        Generate synthetic retail and customer data.
        
        Returns:
            Tuple[str, str]: Paths to transaction and customer data files
        """
        # Generate synthetic transaction data
        np.random.seed(42)
        n_customers = 1000
        n_transactions = 50000
        
        # Generate customer data
        customers = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.randint(18, 80, n_customers),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'income': np.random.normal(50000, 20000, n_customers),
            'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_customers),
            'join_date': [
                datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 365*3))
                for _ in range(n_customers)
            ]
        }
        
        customers_df = pd.DataFrame(customers)
        
        # Generate transaction data
        transactions = {
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.choice(customers_df['customer_id'], n_transactions),
            'transaction_date': [
                datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 365*3))
                for _ in range(n_transactions)
            ],
            'amount': np.random.lognormal(4, 1, n_transactions),  # Log-normal distribution for amounts
            'product_category': np.random.choice(
                ['Electronics', 'Clothing', 'Food', 'Home', 'Beauty'],
                n_transactions
            )
        }
        
        transactions_df = pd.DataFrame(transactions)
        
        # Save to CSV files
        transactions_path = os.path.join(self.raw_dir, 'transactions.csv')
        customers_path = os.path.join(self.raw_dir, 'customers.csv')
        
        transactions_df.to_csv(transactions_path, index=False)
        customers_df.to_csv(customers_path, index=False)
        
        self.logger.info(f"Generated synthetic data: {n_transactions} transactions, {n_customers} customers")
        
        return transactions_path, customers_path

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed transaction and customer data
        """
        # Try to load existing processed data
        processed_transactions = os.path.join(self.processed_dir, 'processed_transactions.csv')
        processed_customers = os.path.join(self.processed_dir, 'processed_customers.csv')
        
        if os.path.exists(processed_transactions) and os.path.exists(processed_customers):
            self.logger.info("Loading preprocessed data...")
            transactions_df = pd.read_csv(processed_transactions)
            customers_df = pd.read_csv(processed_customers)
            
            # Convert date columns
            transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
            customers_df['join_date'] = pd.to_datetime(customers_df['join_date'])
            
            return transactions_df, customers_df
        
        # If no processed data exists, load and process raw data
        self.logger.info("Processing raw data...")
        
        # Try to download data or generate synthetic data
        try:
            transactions_path, customers_path = self.download_kaggle_dataset()
        except:
            transactions_path, customers_path = self.generate_synthetic_data()
        
        # Load raw data
        transactions_df = pd.read_csv(transactions_path)
        customers_df = pd.read_csv(customers_path)
        
        # Preprocess transactions
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        transactions_df = transactions_df.sort_values('transaction_date')
        
        # Remove duplicates and invalid transactions
        transactions_df = transactions_df.drop_duplicates()
        transactions_df = transactions_df[transactions_df['amount'] > 0]
        
        # Preprocess customers
        customers_df['join_date'] = pd.to_datetime(customers_df['join_date'])
        customers_df = customers_df.drop_duplicates(subset=['customer_id'])
        
        # Add derived features
        transactions_df['year_month'] = transactions_df['transaction_date'].dt.to_period('M')
        transactions_df['day_of_week'] = transactions_df['transaction_date'].dt.dayofweek
        
        # Save processed data
        transactions_df.to_csv(processed_transactions, index=False)
        customers_df.to_csv(processed_customers, index=False)
        
        self.logger.info("Data preprocessing complete!")
        
        return transactions_df, customers_df

    def create_sqlite_database(self, 
                             transactions_df: pd.DataFrame, 
                             customers_df: pd.DataFrame) -> None:
        """
        Create SQLite database from processed DataFrames.
        
        Parameters:
            transactions_df (pd.DataFrame): Processed transaction data
            customers_df (pd.DataFrame): Processed customer data
        """
        import sqlite3
        
        db_path = os.path.join(self.processed_dir, 'retail_analysis.db')
        
        # Create database connection
        conn = sqlite3.connect(db_path)
        
        # Save DataFrames to SQLite
        transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)
        
        # Create indices for better performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_customer_id ON transactions(customer_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(transaction_date)')
        
        conn.close()
        
        self.logger.info(f"SQLite database created at {db_path}")

def main():
    """
    Main function to execute data acquisition and processing pipeline.
    """
    # Initialize data acquisition
    da = DataAcquisition()
    
    # Load and process data
    transactions_df, customers_df = da.load_and_preprocess_data()
    
    # Create SQLite database
    da.create_sqlite_database(transactions_df, customers_df)
    
    print("Data acquisition and processing complete!")
    print(f"Transactions shape: {transactions_df.shape}")
    print(f"Customers shape: {customers_df.shape}")
    
    # Display sample of the data
    print("\nSample transactions:")
    print(transactions_df.head())
    print("\nSample customers:")
    print(customers_df.head())

if __name__ == "__main__":
    main()
