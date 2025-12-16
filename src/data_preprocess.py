import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# --- 1. Manual WoE Implementation (Replacing xverse.WOE) ---

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> Tuple[Dict, float, List]:
    """Calculates WoE and IV for a single feature."""
    
    n_unique = df[feature].nunique()
    n_bins = min(10, max(2, n_unique)) 

    if n_bins < 2:
        return {}, 0.0, []

    # 1. Equal Frequency Binning
    try:
        df['bins'] = pd.qcut(df[feature], q=n_bins, duplicates='drop', retbins=False)
    except Exception:
        df['bins'] = pd.cut(df[feature], bins=n_bins, include_lowest=True)
    
    bin_categories = df['bins'].cat.categories.tolist()
    
    # 2. Calculate WoE for each bin
    grouped = df.groupby('bins')[target].agg(['count', 'sum']).reset_index()
    grouped.rename(columns={'sum': 'bads', 'count': 'total'}, inplace=True)
    grouped['goods'] = grouped['total'] - grouped['bads']

    total_goods = grouped['goods'].sum()
    total_bads = grouped['bads'].sum()

    epsilon = 0.0001
    grouped['prop_goods'] = grouped['goods'].apply(lambda x: x / total_goods if total_goods > 0 else 0.0)
    grouped['prop_bads'] = grouped['bads'].apply(lambda x: x / total_bads if total_bads > 0 else 0.0)
    
    # WoE formula
    grouped['woe'] = np.log((grouped['prop_goods'] + epsilon) / (grouped['prop_bads'] + epsilon))
    
    # Calculate IV
    grouped['IV_contribution'] = (grouped['prop_goods'] - grouped['prop_bads']) * grouped['woe']
    IV = grouped['IV_contribution'].sum()
    
    woe_map = dict(zip(grouped['bins'], grouped['woe']))
    
    return woe_map, IV, bin_categories

class WOETransformerManual(BaseEstimator, TransformerMixin):
    """Manually performs WoE transformation for all features using direct indexing."""
    def __init__(self):
        self.woe_maps: Dict[str, Dict] = {}
        self.iv_scores: Dict[str, float] = {}
        self.features: List[str] = []
        self.bin_categories_map: Dict[str, List] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_copy = X.copy()
        X_copy['target'] = y.reset_index(drop=True)
        self.features = X.columns.tolist()

        for feature in self.features:
            woe_map, iv_score, categories = calculate_woe_iv(X_copy[['target', feature]].copy(), feature, 'target')
            self.woe_maps[feature] = woe_map
            self.iv_scores[feature] = iv_score
            self.bin_categories_map[feature] = categories
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = pd.DataFrame(index=X.index)
        
        for feature in self.features:
            
            woe_map = self.woe_maps[feature]
            categories = self.bin_categories_map[feature]
            n_bins = len(categories)
            
            if n_bins < 2:
                X_transformed[f'{feature}_WOE'] = 0.0
                continue
            
            temp_df = X[[feature]].copy()

            # Re-run the binning logic used in fit
            try:
                binned_series = pd.qcut(temp_df[feature], q=n_bins, duplicates='drop', retbins=False)
            except Exception:
                binned_series = pd.cut(temp_df[feature], bins=n_bins, include_lowest=True)

            woe_values = [woe_map[c] for c in categories]
            
            # Use .codes for direct, integer-based indexing to bypass Categorical structure issues.
            # Code -1 means NaN/missing value (out of range), which is mapped to 0.0
            
            woe_array = np.zeros(len(temp_df))
            codes = binned_series.cat.codes.to_numpy()
            
            valid_mask = codes != -1
            woe_array[valid_mask] = np.array(woe_values)[codes[valid_mask]]
            
            X_transformed[f'{feature}_WOE'] = woe_array

        return X_transformed.astype(float)


# --- 2. SUPPORTING TRANSFORMERS (Target and Feature Engineering) ---

class TargetEngineer(BaseEstimator, TransformerMixin):
    """Calculates RFM metrics, performs K-Means clustering, and creates the binary 'is_high_risk' target variable."""
    def __init__(self, date_col='TransactionStartTime', value_col='Value', n_clusters=3, random_state=42):
        self.date_col = date_col
        self.value_col = value_col
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.high_risk_cluster_label = None 

    def fit(self, X, y=None):
        X_copy = X.copy()
        snapshot_date = X_copy[self.date_col].max() + pd.Timedelta(days=1)
        
        rfm_df = X_copy.groupby('CustomerId').agg(
            Recency=(self.date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=(self.value_col, 'sum')
        ).reset_index()

        rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        rfm_df['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Define High-Risk Cluster: highest mean Recency
        cluster_summary = rfm_df.groupby('Cluster')['Recency'].mean()
        self.high_risk_cluster_label = cluster_summary.idxmax()
        
        return self

    def transform(self, X):
        X_copy = X.copy()
        snapshot_date = X_copy[self.date_col].max() + pd.Timedelta(days=1)
        
        rfm_df = X_copy.groupby('CustomerId').agg(
            Recency=(self.date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=(self.value_col, 'sum')
        ).reset_index()

        rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
        rfm_scaled = self.scaler.transform(rfm_features)
        rfm_df['Cluster'] = self.kmeans.predict(rfm_scaled)

        rfm_df['is_high_risk'] = np.where(rfm_df['Cluster'] == self.high_risk_cluster_label, 1, 0)
        
        return rfm_df[['CustomerId', 'is_high_risk']]

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts and aggregates temporal features from TransactionStartTime."""
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col], errors='coerce') 
        
        X_copy['transaction_hour'] = X_copy[self.date_col].dt.hour
        X_copy['transaction_month'] = X_copy[self.date_col].dt.month
        
        temporal_agg = X_copy.groupby('CustomerId').agg(
            avg_transaction_hour=('transaction_hour', 'mean'),
            mode_transaction_month=('transaction_month', lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        ).reset_index()
        
        return temporal_agg

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Calculates key customer-level RFM-related monetary and frequency aggregates."""
    def __init__(self, value_col='Value'):
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        if not np.issubdtype(X_copy[self.value_col].dtype, np.number):
             X_copy[self.value_col] = pd.to_numeric(X_copy[self.value_col], errors='coerce')

        agg_features = X_copy.groupby('CustomerId').agg(
            transaction_count=('TransactionId', 'count'),  
            total_amount=(self.value_col, 'sum'),                 
            average_amount=(self.value_col, 'mean'),              
            std_amount=(self.value_col, 'std')                    
        ).reset_index()

        agg_features['std_amount'] = agg_features['std_amount'].fillna(0)
        
        return agg_features


# --- 3. CORE PIPELINE FUNCTIONS ---

def load_and_clean_data(filepath):
    """Loads data and forces strict type conversion on necessary columns."""
    df = pd.read_csv(filepath)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df.dropna(subset=['Value', 'TransactionStartTime', 'CustomerId'], inplace=True)
    return df

def create_full_data_matrix(raw_data_df: pd.DataFrame):
    """Executes feature engineering and WoE transformation on raw data."""
    
    target_engineer = TargetEngineer()
    target_engineer.fit(raw_data_df)
    target_df = target_engineer.transform(raw_data_df)
    
    agg_df = AggregateFeatures().transform(raw_data_df)
    temp_df = TemporalFeatureExtractor().transform(raw_data_df)
    
    customer_features_df = pd.merge(agg_df, temp_df, on='CustomerId', how='inner')
    full_df = pd.merge(customer_features_df, target_df, on='CustomerId', how='inner')
    
    X = full_df.drop(columns=['CustomerId', 'is_high_risk'])
    y = full_df['is_high_risk']

    # Robust Numeric Conversion
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.astype(float) 

    # WoE Transformation (Manual)
    woe_transformer = WOETransformerManual()
    woe_transformer.fit(X, y)
    X_woe = woe_transformer.transform(X)
    
    return X_woe, y, woe_transformer
