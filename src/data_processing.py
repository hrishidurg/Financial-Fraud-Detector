"""
Data Processing Module for Financial Fraud Detection
Handles data ingestion, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionDataProcessor:
    """Process and engineer features from transaction data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """Load transaction data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} transactions from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df):
        """Clean and prepare data"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna({
            'amount': df['amount'].median(),
            'merchant_category': 'unknown',
            'location': 'unknown'
        })
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Cleaned data: {len(df)} rows")
        return df
    
    def extract_features(self, df):
        """Extract and engineer features for fraud detection"""
        logger.info("Extracting features...")
        
        features_df = pd.DataFrame()
        
        # Basic transaction features
        features_df['amount'] = df['amount']
        features_df['amount_log'] = np.log1p(df['amount'])
        
        # Amount z-score (normalized by user)
        if 'user_id' in df.columns:
            features_df['amount_zscore'] = (
                df['amount'] - df.groupby('user_id')['amount'].transform('mean')
            ) / df.groupby('user_id')['amount'].transform('std')
            features_df['amount_zscore'] = features_df['amount_zscore'].fillna(0)
        
        # Time-based features
        if 'timestamp' in df.columns:
            features_df['hour'] = df['timestamp'].dt.hour
            features_df['day_of_week'] = df['timestamp'].dt.dayofweek
            features_df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            features_df['is_night'] = ((df['timestamp'].dt.hour >= 22) | 
                                       (df['timestamp'].dt.hour <= 6)).astype(int)
        
        # Velocity features (transactions per time window)
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            features_df['transactions_last_hour'] = self._calculate_velocity(
                df_sorted, 'user_id', 'timestamp', hours=1
            )
            features_df['transactions_last_day'] = self._calculate_velocity(
                df_sorted, 'user_id', 'timestamp', hours=24
            )
        
        # Merchant category encoding
        if 'merchant_category' in df.columns:
            features_df['merchant_category_encoded'] = self.label_encoder.fit_transform(
                df['merchant_category'].fillna('unknown')
            )
            
            # High-risk merchant categories
            high_risk_categories = ['gambling', 'crypto', 'money_transfer']
            features_df['is_high_risk_merchant'] = df['merchant_category'].isin(
                high_risk_categories
            ).astype(int)
        
        # Location-based features
        if 'location' in df.columns:
            features_df['location_encoded'] = self.label_encoder.fit_transform(
                df['location'].fillna('unknown')
            )
        
        # Device features
        if 'device_id' in df.columns:
            # New device indicator
            features_df['is_new_device'] = (~df.groupby('user_id')['device_id'].transform(
                lambda x: x.duplicated()
            )).astype(int)
        
        # Distance from previous transaction
        if all(col in df.columns for col in ['latitude', 'longitude', 'user_id', 'timestamp']):
            features_df['distance_from_last'] = self._calculate_distance_velocity(df)
        
        logger.info(f"Extracted {len(features_df.columns)} features")
        return features_df
    
    def _calculate_velocity(self, df, user_col, time_col, hours=1):
        """Calculate transaction velocity (count within time window)"""
        velocity = []
        for idx, row in df.iterrows():
            user_mask = df[user_col] == row[user_col]
            time_mask = (df[time_col] >= row[time_col] - timedelta(hours=hours)) & \
                       (df[time_col] <= row[time_col])
            count = df[user_mask & time_mask].shape[0]
            velocity.append(count)
        return velocity
    
    def _calculate_distance_velocity(self, df):
        """Calculate distance from previous transaction"""
        distances = []
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        
        for user_id in df_sorted['user_id'].unique():
            user_df = df_sorted[df_sorted['user_id'] == user_id].reset_index(drop=True)
            user_distances = [0]  # First transaction has 0 distance
            
            for i in range(1, len(user_df)):
                dist = self._haversine_distance(
                    user_df.loc[i-1, 'latitude'], user_df.loc[i-1, 'longitude'],
                    user_df.loc[i, 'latitude'], user_df.loc[i, 'longitude']
                )
                user_distances.append(dist)
            
            distances.extend(user_distances)
        
        return distances
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def prepare_training_data(self, features_df, labels):
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        # Scale numerical features
        scaled_features = self.scaler.fit_transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
        
        return scaled_df, labels
    
    def prepare_inference_data(self, transaction_dict):
        """Prepare single transaction for real-time inference"""
        # Convert dictionary to DataFrame
        df = pd.DataFrame([transaction_dict])
        
        # Extract features
        features = self.extract_features(df)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        return scaled_features


def generate_sample_data(num_transactions=10000, fraud_ratio=0.02):
    """Generate synthetic transaction data for testing"""
    logger.info(f"Generating {num_transactions} sample transactions...")
    
    np.random.seed(42)
    
    # Generate user IDs
    num_users = 1000
    user_ids = np.random.choice(range(1, num_users + 1), num_transactions)
    
    # Generate timestamps
    base_time = datetime.now() - timedelta(days=30)
    timestamps = [base_time + timedelta(seconds=np.random.randint(0, 30*24*60*60)) 
                  for _ in range(num_transactions)]
    
    # Generate transaction amounts
    amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=num_transactions)
    amounts = np.clip(amounts, 1, 10000)
    
    # Generate merchant categories
    categories = np.random.choice(
        ['grocery', 'restaurant', 'retail', 'gas', 'online', 'gambling', 'crypto'],
        num_transactions,
        p=[0.3, 0.2, 0.25, 0.1, 0.1, 0.03, 0.02]
    )
    
    # Generate locations
    locations = np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'CN', 'RU'],
                                 num_transactions,
                                 p=[0.5, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
    
    # Generate device IDs
    device_ids = np.random.choice(range(1, num_users * 3), num_transactions)
    
    # Generate lat/long
    latitudes = np.random.uniform(-90, 90, num_transactions)
    longitudes = np.random.uniform(-180, 180, num_transactions)
    
    # Generate fraud labels
    num_fraud = int(num_transactions * fraud_ratio)
    is_fraud = np.array([1] * num_fraud + [0] * (num_transactions - num_fraud))
    np.random.shuffle(is_fraud)
    
    # Create DataFrame
    data = pd.DataFrame({
        'transaction_id': range(1, num_transactions + 1),
        'user_id': user_ids,
        'timestamp': timestamps,
        'amount': amounts,
        'merchant_category': categories,
        'location': locations,
        'device_id': device_ids,
        'latitude': latitudes,
        'longitude': longitudes,
        'is_fraud': is_fraud
    })
    
    logger.info(f"Generated {len(data)} transactions with {is_fraud.sum()} fraudulent cases")
    return data


if __name__ == "__main__":
    # Example usage
    processor = TransactionDataProcessor()
    
    # Generate sample data
    sample_data = generate_sample_data(10000, fraud_ratio=0.02)
    sample_data.to_csv('sample_transactions.csv', index=False)
    
    # Load and process data
    df = processor.load_data('sample_transactions.csv')
    df = processor.clean_data(df)
    features = processor.extract_features(df)
    
    print("\nFeatures extracted:")
    print(features.head())
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"Shape: {features.shape}")
