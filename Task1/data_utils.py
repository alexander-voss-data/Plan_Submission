"""
Data loading and inspection utilities for Trial Activation Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_and_inspect_data(file_path):
    """
    Load CSV data and perform initial inspection.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Basic data inspection
        logger.info("=== DATA OVERVIEW ===")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for any obvious issues
        logger.info(f"Null values per column:\n{df.isnull().sum()}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def display_basic_info(df):
    """
    Display comprehensive basic information about the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
    """
    logger.info("=== BASIC DATASET INFORMATION ===")
    
    # Shape and basic info
    logger.info(f"Dataset dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    logger.info("\n=== COLUMN INFORMATION ===")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        logger.info(f"{i:2d}. {col:30} | {str(dtype):15} | "
                   f"Null: {null_count:6,} ({null_pct:5.1f}%) | "
                   f"Unique: {unique_count:8,}")
    
    # Sample data
    logger.info("\n=== SAMPLE DATA (First 5 rows) ===")
    logger.info(f"\n{df.head().to_string()}")
    
    # Data type summary
    logger.info(f"\n=== DATA TYPE SUMMARY ===")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"{str(dtype):15}: {count:3d} columns")
    
    return df
