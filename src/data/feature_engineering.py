"""
AI-Powered Financial Fraud Detection System
Feature Engineering Module

This module implements feature engineering for the fraud detection system,
including temporal features, aggregations, and transformations.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Local imports
from ..utils.logger import get_logger
from ..config.model_config import FEATURE_ENGINEERING_PARAMS

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for fraud detection.
    
    This class implements various feature engineering techniques:
    1. Temporal features (hour, day, month, etc.)
    2. Aggregation features (transaction history)
    3. Behavioral features (customer patterns)
    4. Geographical features (location-based)
    5. Network features (connections between entities)
    """
    
    def __init__(
        self,
        time_window_sizes: List[int] = None,
        aggregation_functions: List[str] = None,
        categorical_encoding: str = 'one-hot',
        numerical_scaling: str = 'standard',
        feature_selection_threshold: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize the feature engineer.
        
        Args:
            time_window_sizes: List of time windows in days for aggregations
            aggregation_functions: List of aggregation functions to apply
            categorical_encoding: Method for encoding categorical variables
            numerical_scaling: Method for scaling numerical variables
            feature_selection_threshold: Threshold for feature importance
            random_state: Random seed for reproducibility
        """
        # Initialize parameters with defaults if not provided
        self.time_window_sizes = time_window_sizes or FEATURE_ENGINEERING_PARAMS['time_window_sizes']
        self.aggregation_functions = aggregation_functions or FEATURE_ENGINEERING_PARAMS['aggregation_functions']
        self.categorical_encoding = categorical_encoding or FEATURE_ENGINEERING_PARAMS['categorical_encoding']
        self.numerical_scaling = numerical_scaling or FEATURE_ENGINEERING_PARAMS['numerical_scaling']
        self.feature_selection_threshold = feature_selection_threshold or FEATURE_ENGINEERING_PARAMS['feature_selection_threshold']
        self.random_state = random_state
        
        # Initialize transformers
        self.numerical_scaler = None
        if self.numerical_scaling == 'standard':
            self.numerical_scaler = StandardScaler()
        elif self.numerical_scaling == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        
        self.categorical_encoder = None
        if self.categorical_encoding == 'one-hot':
            self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        self.feature_selector = None
        
        logger.info("Initialized FeatureEngineer")
    
    def fit(
        self,
        transactions: pd.DataFrame,
        labels: pd.Series,
        numerical_columns: List[str] = None,
        categorical_columns: List[str] = None,
        datetime_column: str = 'timestamp',
        customer_id_column: str = 'customer_id',
        merchant_id_column: str = 'merchant_id'
    ) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline on training data.
        
        Args:
            transactions: DataFrame of transactions
            labels: Series of fraud labels (0 for legitimate, 1 for fraud)
            numerical_columns: List of numerical columns
            categorical_columns: List of categorical columns
            datetime_column: Name of the datetime column
            customer_id_column: Name of the customer ID column
            merchant_id_column: Name of the merchant ID column
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting feature engineering pipeline on {len(transactions)} transactions")
        
        # Infer column types if not provided
        if numerical_columns is None:
            numerical_columns = transactions.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Exclude ID columns
            numerical_columns = [col for col in numerical_columns if 'id' not in col.lower()]
        
        if categorical_columns is None:
            categorical_columns = transactions.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude datetime column
            if datetime_column in categorical_columns:
                categorical_columns.remove(datetime_column)
        
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.datetime_column = datetime_column
        self.customer_id_column = customer_id_column
        self.merchant_id_column = merchant_id_column
        
        # Create a copy of the data
        df = transactions.copy()
        
        # Generate temporal features
        df = self._generate_temporal_features(df)
        
        # Generate aggregation features
        df = self._generate_aggregation_features(df)
        
        # Fit numerical scaler
        if self.numerical_scaler is not None:
            numerical_data = df[self.numerical_columns].fillna(0)
            self.numerical_scaler.fit(numerical_data)
            logger.info(f"Fitted numerical scaler on {len(numerical_data.columns)} columns")
        
        # Fit categorical encoder
        if self.categorical_encoder is not None and self.categorical_columns:
            categorical_data = df[self.categorical_columns].fillna('missing')
            self.categorical_encoder.fit(categorical_data)
            logger.info(f"Fitted categorical encoder on {len(categorical_data.columns)} columns")
        
        # Apply transformations to get the full feature set
        df_transformed = self._transform_features(df)
        
        # Fit feature selector
        if self.feature_selection_threshold > 0:
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                ),
                threshold=self.feature_selection_threshold
            )
            self.feature_selector.fit(df_transformed, labels)
            
            # Get selected feature indices
            selected_features = self.feature_selector.get_support()
            n_selected = sum(selected_features)
            logger.info(f"Selected {n_selected} features out of {len(selected_features)}")
        
        return self
    
    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transactions data using the fitted feature engineering pipeline.
        
        Args:
            transactions: DataFrame of transactions
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Transforming {len(transactions)} transactions")
        
        # Create a copy of the data
        df = transactions.copy()
        
        # Generate temporal features
        df = self._generate_temporal_features(df)
        
        # Generate aggregation features
        df = self._generate_aggregation_features(df)
        
        # Apply transformations
        df_transformed = self._transform_features(df)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            df_transformed = self.feature_selector.transform(df_transformed)
            logger.info(f"Applied feature selection, resulting in {df_transformed.shape[1]} features")
        
        return df_transformed
    
    def fit_transform(
        self,
        transactions: pd.DataFrame,
        labels: pd.Series,
        numerical_columns: List[str] = None,
        categorical_columns: List[str] = None,
        datetime_column: str = 'timestamp',
        customer_id_column: str = 'customer_id',
        merchant_id_column: str = 'merchant_id'
    ) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data.
        
        Args:
            transactions: DataFrame of transactions
            labels: Series of fraud labels (0 for legitimate, 1 for fraud)
            numerical_columns: List of numerical columns
            categorical_columns: List of categorical columns
            datetime_column: Name of the datetime column
            customer_id_column: Name of the customer ID column
            merchant_id_column: Name of the merchant ID column
            
        Returns:
            DataFrame with engineered features
        """
        self.fit(
            transactions,
            labels,
            numerical_columns,
            categorical_columns,
            datetime_column,
            customer_id_column,
            merchant_id_column
        )
        return self.transform(transactions)
    
    def _generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate temporal features from datetime column.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            DataFrame with added temporal features
        """
        if self.datetime_column not in df.columns:
            logger.warning(f"Datetime column '{self.datetime_column}' not found in data")
            return df
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        
        # Extract temporal features
        df['hour_of_day'] = df[self.datetime_column].dt.hour
        df['day_of_week'] = df[self.datetime_column].dt.dayofweek
        df['day_of_month'] = df[self.datetime_column].dt.day
        df['month'] = df[self.datetime_column].dt.month
        df['quarter'] = df[self.datetime_column].dt.quarter
        df['year'] = df[self.datetime_column].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
        
        # Business hours (9 AM to 5 PM on weekdays)
        df['is_business_hours'] = (
            (df['hour_of_day'] >= 9) &
            (df['hour_of_day'] <= 17) &
            (df['day_of_week'] < 5)
        ).astype(int)
        
        logger.info("Generated temporal features")
        return df
    
    def _generate_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate aggregation features based on transaction history.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            DataFrame with added aggregation features
        """
        # Check if required columns exist
        required_columns = [self.datetime_column, self.customer_id_column, self.merchant_id_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for aggregation: {missing_columns}")
            return df
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        
        # Sort by customer and timestamp
        df = df.sort_values([self.customer_id_column, self.datetime_column])
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Numerical columns to aggregate
        agg_columns = [col for col in self.numerical_columns if col in df.columns]
        
        if not agg_columns:
            logger.warning("No numerical columns available for aggregation")
            return df
        
        # Generate customer-level aggregations for different time windows
        for window_days in self.time_window_sizes:
            logger.info(f"Generating aggregations for {window_days}-day window")
            
            # Calculate window start time for each transaction
            window_start = df[self.datetime_column] - timedelta(days=window_days)
            
            # Group by customer and calculate aggregations
            for col in agg_columns:
                for func in self.aggregation_functions:
                    # Skip inappropriate aggregations
                    if func == 'sum' and 'ratio' in col.lower():
                        continue
                    
                    feature_name = f"{col}_{func}_{window_days}d"
                    
                    # Calculate aggregation for each transaction
                    result_df[feature_name] = np.nan
                    
                    # This is a simplified implementation for demonstration
                    # In a real system, use more efficient methods like rolling windows
                    for i, row in df.iterrows():
                        customer = row[self.customer_id_column]
                        current_time = row[self.datetime_column]
                        
                        # Get historical transactions for this customer in the time window
                        mask = (
                            (df[self.customer_id_column] == customer) &
                            (df[self.datetime_column] < current_time) &
                            (df[self.datetime_column] >= current_time - timedelta(days=window_days))
                        )
                        
                        history = df.loc[mask, col]
                        
                        if len(history) > 0:
                            if func == 'mean':
                                result_df.at[i, feature_name] = history.mean()
                            elif func == 'std':
                                result_df.at[i, feature_name] = history.std() if len(history) > 1 else 0
                            elif func == 'min':
                                result_df.at[i, feature_name] = history.min()
                            elif func == 'max':
                                result_df.at[i, feature_name] = history.max()
                            elif func == 'sum':
                                result_df.at[i, feature_name] = history.sum()
                            elif func == 'count':
                                result_df.at[i, feature_name] = len(history)
        
        # Fill missing values with 0
        agg_columns = [col for col in result_df.columns if any(f"_{window}d" in col for window in self.time_window_sizes)]
        result_df[agg_columns] = result_df[agg_columns].fillna(0)
        
        logger.info(f"Generated {len(agg_columns)} aggregation features")
        return result_df
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to features.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            DataFrame with transformed features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply numerical scaling
        if self.numerical_scaler is not None:
            numerical_cols = [col for col in self.numerical_columns if col in result_df.columns]
            if numerical_cols:
                result_df[numerical_cols] = self.numerical_scaler.transform(result_df[numerical_cols].fillna(0))
                logger.info(f"Applied numerical scaling to {len(numerical_cols)} columns")
        
        # Apply categorical encoding
        if self.categorical_encoder is not None and self.categorical_columns:
            categorical_cols = [col for col in self.categorical_columns if col in result_df.columns]
            if categorical_cols:
                # Get encoded feature names
                encoded_features = self.categorical_encoder.get_feature_names_out(categorical_cols)
                
                # Transform categorical columns
                encoded_data = self.categorical_encoder.transform(result_df[categorical_cols].fillna('missing'))
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=encoded_features,
                    index=result_df.index
                )
                
                # Drop original categorical columns and add encoded ones
                result_df = result_df.drop(columns=categorical_cols)
                result_df = pd.concat([result_df, encoded_df], axis=1)
                
                logger.info(f"Applied categorical encoding, resulting in {len(encoded_features)} features")
        
        # Drop non-feature columns
        columns_to_drop = [
            self.datetime_column,
            self.customer_id_column,
            self.merchant_id_column,
            'transaction_id'
        ]
        
        for col in columns_to_drop:
            if col in result_df.columns:
                result_df = result_df.drop(columns=col)
        
        logger.info(f"Final feature set has {result_df.shape[1]} features")
        return result_df
    
    def save(self, path: str) -> None:
        """
        Save the feature engineering pipeline to disk.
        
        Args:
            path: Directory to save the pipeline
        """
        os.makedirs(path, exist_ok=True)
        
        # Save transformers
        if self.numerical_scaler is not None:
            joblib.dump(self.numerical_scaler, os.path.join(path, 'numerical_scaler.pkl'))
        
        if self.categorical_encoder is not None:
            joblib.dump(self.categorical_encoder, os.path.join(path, 'categorical_encoder.pkl'))
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, os.path.join(path, 'feature_selector.pkl'))
        
        # Save configuration
        config = {
            'time_window_sizes': self.time_window_sizes,
            'aggregation_functions': self.aggregation_functions,
            'categorical_encoding': self.categorical_encoding,
            'numerical_scaling': self.numerical_scaling,
            'feature_selection_threshold': self.feature_selection_threshold,
            'random_state': self.random_state,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_column': self.datetime_column,
            'customer_id_column': self.customer_id_column,
            'merchant_id_column': self.merchant_id_column
        }
        
        joblib.dump(config, os.path.join(path, 'feature_engineering_config.pkl'))
        logger.info(f"Feature engineering pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """
        Load the feature engineering pipeline from disk.
        
        Args:
            path: Directory to load the pipeline from
            
        Returns:
            Loaded FeatureEngineer instance
        """
        # Load configuration
        config_path = os.path.join(path, 'feature_engineering_config.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        config = joblib.load(config_path)
        
        # Create instance with loaded configuration
        instance = cls(
            time_window_sizes=config['time_window_sizes'],
            aggregation_functions=config['aggregation_functions'],
            categorical_encoding=config['categorical_encoding'],
            numerical_scaling=config['numerical_scaling'],
            feature_selection_threshold=config['feature_selection_threshold'],
            random_state=config['random_state']
        )
        
        # Set additional attributes
        instance.numerical_columns = config['numerical_columns']
        instance.categorical_columns = config['categorical_columns']
        instance.datetime_column = config['datetime_column']
        instance.customer_id_column = config['customer_id_column']
        instance.merchant_id_column = config['merchant_id_column']
        
        # Load transformers
        numerical_scaler_path = os.path.join(path, 'numerical_scaler.pkl')
        if os.path.exists(numerical_scaler_path):
            instance.numerical_scaler = joblib.load(numerical_scaler_path)
        
        categorical_encoder_path = os.path.join(path, 'categorical_encoder.pkl')
        if os.path.exists(categorical_encoder_path):
            instance.categorical_encoder = joblib.load(categorical_encoder_path)
        
        feature_selector_path = os.path.join(path, 'feature_selector.pkl')
        if os.path.exists(feature_selector_path):
            instance.feature_selector = joblib.load(feature_selector_path)
        
        logger.info(f"Feature engineering pipeline loaded from {path}")
        return instance


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 6, 1)
    timestamps = [
        start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(n_samples)
    ]
    
    # Generate customer IDs
    customer_ids = [f"CUST{i:04d}" for i in np.random.randint(1, 101, n_samples)]
    
    # Generate merchant IDs
    merchant_ids = [f"MERCH{i:03d}" for i in np.random.randint(1, 21, n_samples)]
    
    # Generate transaction amounts
    amounts = np.random.exponential(scale=100, size=n_samples)
    
    # Generate payment methods
    payment_methods = np.random.choice(
        ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'],
        size=n_samples,
        p=[0.5, 0.3, 0.1, 0.1]
    )
    
    # Generate fraud labels (imbalanced)
    fraud_labels = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    
    # Create DataFrame
    data = pd.DataFrame({
        'transaction_id': [f"TX{i:06d}" for i in range(n_samples)],
        'timestamp': timestamps,
        'customer_id': customer_ids,
        'merchant_id': merchant_ids,
        'amount': amounts,
        'payment_method': payment_methods,
        'is_fraud': fraud_labels
    })
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Split data
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and fit feature engineer
    feature_engineer = FeatureEngineer()
    X_train_transformed = feature_engineer.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_transformed = feature_engineer.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Transformed features: {X_train_transformed.shape[1]}")
    
    # Save and load
    feature_engineer.save("models/feature_engineering")
    loaded_engineer = FeatureEngineer.load("models/feature_engineering")
    
    # Verify loaded pipeline
    X_test_transformed_loaded = loaded_engineer.transform(X_test)
    print(f"Loaded pipeline features: {X_test_transformed_loaded.shape[1]}")

