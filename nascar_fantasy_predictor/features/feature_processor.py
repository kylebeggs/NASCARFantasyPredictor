"""Feature processing and cleaning for model training."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path


class FeatureProcessor:
    """Process and clean features for model training."""
    
    def __init__(self):
        self.label_encoders = {}
        self.numeric_features = None
        self.categorical_features = None
    
    def prepare_features_for_training(self, df: pd.DataFrame, 
                                    target_column: str = 'fantasy_points') -> pd.DataFrame:
        """Prepare feature dataframe for model training."""
        
        # Define which columns to exclude
        exclude_columns = [
            'driver_id', 'driver_name', 'race_id', 'date', target_column,
            'team'  # Exclude team for now, too many unique values
        ]
        
        # Define categorical columns that need encoding
        categorical_columns = ['manufacturer']  # Start with just manufacturer for now
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Fill missing values
        processed_df = processed_df.fillna(0)
        
        # Encode categorical features
        for col in categorical_columns:
            if col in processed_df.columns:
                processed_df[col] = self._encode_categorical(col, processed_df[col])
        
        # Select only numeric features for training
        feature_columns = []
        for col in processed_df.columns:
            if col not in exclude_columns:
                # Check if column is numeric or can be converted
                try:
                    pd.to_numeric(processed_df[col])
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    print(f"Skipping non-numeric column: {col}")
                    continue
        
        # Add encoded categorical features
        for col in categorical_columns:
            if col in processed_df.columns:
                feature_columns.append(col)
        
        self.numeric_features = feature_columns
        
        # Return only the feature columns plus target
        if target_column in processed_df.columns:
            return processed_df[feature_columns + [target_column]]
        else:
            return processed_df[feature_columns]
    
    def _encode_categorical(self, column_name: str, series: pd.Series) -> pd.Series:
        """Encode categorical column with label encoder."""
        
        # Convert to string and handle missing values
        series_clean = series.astype(str).fillna('Unknown')
        
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            # Fit the encoder
            encoded = self.label_encoders[column_name].fit_transform(series_clean)
        else:
            # Transform using existing encoder, handle unseen values
            encoder = self.label_encoders[column_name]
            encoded = []
            for value in series_clean:
                try:
                    encoded.append(encoder.transform([value])[0])
                except ValueError:
                    # Unseen value, assign a default (0 or last known value)
                    encoded.append(0)
            encoded = np.array(encoded)
        
        return pd.Series(encoded, index=series.index)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return self.numeric_features or []
    
    def save_encoders(self, filepath: str):
        """Save label encoders to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'numeric_features': self.numeric_features
            }, f)
    
    def load_encoders(self, filepath: str):
        """Load label encoders from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['label_encoders']
            self.numeric_features = data['numeric_features']


def clean_feature_dataframe(df: pd.DataFrame, target_column: str = 'fantasy_points') -> pd.DataFrame:
    """Quick utility function to clean features for training."""
    processor = FeatureProcessor()
    return processor.prepare_features_for_training(df, target_column)