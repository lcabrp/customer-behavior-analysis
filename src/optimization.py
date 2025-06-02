# src/optimization.py
import pandas as pd
import numpy as np
from typing import List
import dask.dataframe as dd
from functools import lru_cache

class PerformanceOptimizer:
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        """
        numerics = ['int16', 'int32', 'int64', 'float64']
        
        for col in df.select_dtypes(include=numerics).columns:
            col_type = df[col].dtype
            
            if col_type in ['int64', 'float64']:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if col_type == 'int64':
                    if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    @staticmethod
    def parallel_process(df: pd.DataFrame, 
                        func: callable, 
                        partition_size: int = 100000) -> pd.DataFrame:
        """
        Process large DataFrames in parallel using Dask.
        """
        ddf = dd.from_pandas(df, npartitions=len(df) // partition_size + 1)
        result = ddf.map_partitions(func).compute()
        return result
    
    @staticmethod
    @lru_cache(maxsize=128)
    def cached_calculation(key: str, data: tuple) -> float:
        """
        Cache expensive calculations.
        """
        # Convert tuple to numpy array for calculations
        arr = np.array(data)
        
        if key == 'mean':
            return float(arr.mean())
        elif key == 'std':
            return float(arr.std())
        elif key == 'percentile':
            return float(np.percentile(arr, 95))
        
        return 0.0
