import os 
import sys


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from data_preprocess import load_and_clean_data, TargetEngineer, AggregateFeatures, TemporalFeatureExtractor

def execute_target_engineering(filepath: str):
    """
    Executes the RFM calculation, K-Means clustering, and high-risk target assignment.
    """
    
    print("--- Starting Task 4: Proxy Target Variable Engineering ---")

    try:
        raw_data = load_and_clean_data(filepath)
        print(f"Data loaded and cleaned. Total transactions: {raw_data.shape[0]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Instantiate and Fit TargetEngineer
    # The TargetEngineer class handles RFM calculation, scaling, K-Means (n_clusters=3), 
    # and identifying the high-risk cluster (highest Recency).
    target_engineer = TargetEngineer(n_clusters=3, random_state=42)
    
    # Fit: This performs RFM calculation, scaling, and K-Means clustering to identify the high-risk cluster label.
    target_engineer.fit(raw_data)
    
    # Transform: This uses the fitted cluster centers to assign a high-risk label (1 or 0)
    # to every unique CustomerId.
    target_df = target_engineer.transform(raw_data)
    
    print(f"\nTarget variable created for {target_df.shape[0]} unique customers.")
    
    # 3. Analyze and Display Results
    
    # Analyze Cluster Characteristics (Requires accessing internal state for analysis)
    # Note: Accessing internal attributes like .kmeans is done here for reporting, 
    # but is generally avoided in production code.
    
    cluster_means = target_engineer.scaler.inverse_transform(target_engineer.kmeans.cluster_centers_)
    cluster_labels = np.arange(target_engineer.n_clusters)
    
    analysis_df = pd.DataFrame(cluster_means, 
                               columns=['Mean_Recency', 'Mean_Frequency', 'Mean_Monetary'],
                               index=[f'Cluster {i}' for i in cluster_labels])
    
    analysis_df['High_Risk_Label'] = np.where(cluster_labels == target_engineer.high_risk_cluster_label, 'YES (Target=1)', 'NO (Target=0)')
    
    # Calculate the size of each segment
    risk_counts = target_df['is_high_risk'].value_counts()
    
    print("\n--- K-Means Cluster Analysis (RFM) ---")
    print("The High-Risk cluster is defined as the one with the HIGHEST Mean_Recency.")
    print(analysis_df)
    
    print("\n--- Target Variable Distribution ---")
    print(f"High Risk (1): {risk_counts.get(1, 0)} customers ({risk_counts.get(1, 0) / target_df.shape[0] * 100:.2f}%)")
    print(f"Low Risk (0): {risk_counts.get(0, 0)} customers ({risk_counts.get(0, 0) / target_df.shape[0] * 100:.2f}%)")

    # 4. Integrate the Target Variable (Demonstration)
    
    # We will also run the feature engineering functions to show the final matrix assembly
    agg_df = AggregateFeatures().transform(raw_data)
    temp_df = TemporalFeatureExtractor().transform(raw_data)
    
    customer_features_df = pd.merge(agg_df, temp_df, on='CustomerId', how='inner')
    final_data_matrix = pd.merge(customer_features_df, target_df, on='CustomerId', how='inner')
    
    print(f"\nIntegrated Data Matrix Shape (Features + Target): {final_data_matrix.shape}")
    print("\nSample of Final Data Matrix (Features + Target):")
    print(final_data_matrix[['CustomerId', 'total_amount', 'avg_transaction_hour', 'is_high_risk']].head())
    
    print("\n--- Task 4 Complete ---")


if __name__ == '__main__':
    data_file_path = 'Data/data.csv' 
    execute_target_engineering(data_file_path)
