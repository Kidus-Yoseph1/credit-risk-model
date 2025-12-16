import os 
import sys
import pytest
import pandas as pd
from datetime import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from src.data_preprocess import AggregateFeatures, TemporalFeatureExtractor

# Sample data for testing (minimal required columns)
@pytest.fixture
def sample_raw_data():
    """Provides a sample DataFrame simulating raw transaction data."""
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 201, 202, 301],
        'Value': [100.0, 50.0, 200.0, 200.0, 300.0],
        'TransactionStartTime': ['2023-01-15 10:00:00', '2023-01-15 11:00:00', 
                                 '2023-02-01 12:00:00', '2023-02-01 13:00:00', 
                                 '2023-03-10 14:00:00']
    }
    df = pd.DataFrame(data)
    # Ensure dtypes match expected input for the transformers
    df['Value'] = pd.to_numeric(df['Value'])
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# --- Unit Test 1: AggregateFeatures ---
def test_aggregate_features_columns(sample_raw_data):
    """Test that AggregateFeatures transformer returns the expected column names."""
    agg_transformer = AggregateFeatures(value_col='Value')
    result_df = agg_transformer.transform(sample_raw_data)
    
    expected_columns = {'CustomerId', 'transaction_count', 'total_amount', 
                        'average_amount', 'std_amount'}
    
    assert set(result_df.columns) == expected_columns
    assert result_df.shape[0] == sample_raw_data['CustomerId'].nunique()
    
def test_aggregate_features_values(sample_raw_data):
    """Test the calculation accuracy of AggregateFeatures."""
    agg_transformer = AggregateFeatures(value_col='Value')
    result_df = agg_transformer.transform(sample_raw_data)
    
    # Test Customer 1 (100 + 50 = 150, count=2, avg=75, std=sqrt( (25^2 + (-25)^2) / 1) = 35.35)
    c1_row = result_df[result_df['CustomerId'] == 1].iloc[0]
    assert c1_row['transaction_count'] == 2
    assert c1_row['total_amount'] == 150.0
    assert c1_row['average_amount'] == 75.0
    assert np.isclose(c1_row['std_amount'], 35.355339)
    
    # Test Customer 3 (count=1, total=300, std=0)
    c3_row = result_df[result_df['CustomerId'] == 3].iloc[0]
    assert c3_row['std_amount'] == 0.0

# --- Unit Test 2: TemporalFeatureExtractor ---
def test_temporal_features_columns(sample_raw_data):
    """Test that TemporalFeatureExtractor returns the expected column names."""
    temp_transformer = TemporalFeatureExtractor()
    result_df = temp_transformer.transform(sample_raw_data)
    
    expected_columns = {'CustomerId', 'avg_transaction_hour', 'mode_transaction_month'}
    
    assert set(result_df.columns) == expected_columns

def test_temporal_features_values(sample_raw_data):
    """Test the calculation accuracy of TemporalFeatureExtractor."""
    temp_transformer = TemporalFeatureExtractor()
    result_df = temp_transformer.transform(sample_raw_data)
    
    # Test Customer 2 (Hours: 12, 13. Months: 2, 2)
    c2_row = result_df[result_df['CustomerId'] == 2].iloc[0]
    assert c2_row['avg_transaction_hour'] == 12.5
    assert c2_row['mode_transaction_month'] == 2.0
    
    # Test Customer 1 (Hours: 10, 11. Months: 1, 1)
    c1_row = result_df[result_df['CustomerId'] == 1].iloc[0]
    assert c1_row['avg_transaction_hour'] == 10.5
    assert c1_row['mode_transaction_month'] == 1.0
