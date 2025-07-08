"""Simplified feature engineering for NASCAR fantasy predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder

from ..data.csv_manager import CSVDataManager


class FeatureEngineer:
    """Engineer essential features for NASCAR fantasy point prediction."""
    
    def __init__(self, data_manager: CSVDataManager):
        self.data_manager = data_manager
        self.label_encoders = {}
        self.feature_columns = None
    
    def create_features(self, target_race_date: str, lookback_races: int = 10) -> pd.DataFrame:
        """Create feature matrix for all drivers for target race."""
        active_drivers = self.data_manager.get_active_drivers(target_race_date)
        
        features = []
        for driver_name in active_drivers:
            driver_features = self._create_driver_features(
                driver_name, target_race_date, lookback_races
            )
            features.append(driver_features)
        
        return pd.DataFrame(features)
    
    def _create_driver_features(self, driver_name: str, target_race_date: str, lookback_races: int) -> Dict:
        """Create essential feature set for a single driver."""
        features = {'driver_name': driver_name}
        
        # Get driver's recent results
        recent_results = self.data_manager.get_driver_recent_results(
            driver_name, target_race_date, lookback_races
        )
        
        if recent_results.empty:
            return self._create_default_features(driver_name)
        
        # Core performance metrics
        features.update(self._calculate_performance_metrics(recent_results))
        
        # Track-specific features
        features.update(self._calculate_track_features(driver_name, target_race_date))
        
        # Recent form
        features.update(self._calculate_recent_form(recent_results))
        
        return features
    
    def _create_default_features(self, driver_name: str) -> Dict:
        """Create default features for drivers with no recent history."""
        return {
            'driver_name': driver_name,
            'avg_finish': 25.0,
            'avg_start': 25.0,
            'races_completed': 0,
            'dnf_rate': 0.2,
            'top_10_rate': 0.1,
            'track_races': 0,
            'track_avg_finish': 25.0,
            'recent_avg_finish': 25.0,
            'form_score': 0.0
        }
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate core performance statistics."""
        features = {}
        
        # Finish position stats
        finish_positions = results['finish_position'].dropna()
        if not finish_positions.empty:
            features['avg_finish'] = finish_positions.mean()
            features['best_finish'] = finish_positions.min()
            features['worst_finish'] = finish_positions.max()
        else:
            features['avg_finish'] = 25.0
            features['best_finish'] = 40
            features['worst_finish'] = 40
        
        # Start position stats
        start_positions = results['start_position'].dropna()
        features['avg_start'] = start_positions.mean() if not start_positions.empty else 25.0
        
        # Race completion stats
        features['races_completed'] = len(results)
        
        # DNF rate
        dnf_count = results['status'].str.upper().str.contains('DNF|ACCIDENT|ENGINE', na=False).sum()
        features['dnf_rate'] = dnf_count / len(results) if len(results) > 0 else 0.2
        
        # Top finishes
        if not finish_positions.empty:
            features['top_10_rate'] = (finish_positions <= 10).mean()
            features['top_5_rate'] = (finish_positions <= 5).mean()
        else:
            features['top_10_rate'] = 0.1
            features['top_5_rate'] = 0.05
        
        return features
    
    def _calculate_track_features(self, driver_name: str, target_race_date: str) -> Dict:
        """Calculate basic track-specific features."""
        # For now, just use overall stats - track detection can be improved later
        return {
            'track_races': 0,
            'track_avg_finish': 25.0
        }
    
    def _calculate_recent_form(self, results: pd.DataFrame) -> Dict:
        """Calculate recent form indicators."""
        features = {}
        
        if len(results) < 2:
            features['recent_avg_finish'] = 25.0
            features['form_score'] = 0.0
            return features
        
        # Sort by date to ensure proper chronological order
        results = results.sort_values('date')
        
        finish_positions = results['finish_position'].dropna()
        if len(finish_positions) >= 3:
            # Recent 3 races vs overall average
            recent_3_avg = finish_positions.iloc[-3:].mean()
            overall_avg = finish_positions.mean()
            features['recent_avg_finish'] = recent_3_avg
            features['form_score'] = overall_avg - recent_3_avg  # Positive = improving
        else:
            features['recent_avg_finish'] = finish_positions.mean() if not finish_positions.empty else 25.0
            features['form_score'] = 0.0
        
        return features
    
    def prepare_for_training(self, df: pd.DataFrame, target_column: str = 'finish_position') -> pd.DataFrame:
        """Prepare feature dataframe for model training."""
        # Define columns to exclude
        exclude_columns = ['driver_id', 'driver_name', 'race_id', 'date', 'team', 'fantasy_points']
        if target_column is not None:
            exclude_columns.append(target_column)
        
        # Make a copy and fill missing values
        processed_df = df.copy()
        processed_df = processed_df.fillna(0)
        
        # Encode categorical features (manufacturer if present)
        if 'manufacturer' in processed_df.columns:
            processed_df['manufacturer'] = self._encode_categorical('manufacturer', processed_df['manufacturer'])
        
        # Select only numeric features for training
        feature_columns = []
        for col in processed_df.columns:
            if col not in exclude_columns:
                try:
                    pd.to_numeric(processed_df[col])
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    continue
        
        self.feature_columns = feature_columns
        
        # Return feature columns plus target if specified
        if target_column in processed_df.columns:
            return processed_df[feature_columns + [target_column]]
        else:
            return processed_df[feature_columns]
    
    def _encode_categorical(self, column_name: str, series: pd.Series) -> pd.Series:
        """Encode categorical column with label encoder."""
        series_clean = series.astype(str).fillna('Unknown')
        
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            encoded = self.label_encoders[column_name].fit_transform(series_clean)
        else:
            encoder = self.label_encoders[column_name]
            encoded = []
            for value in series_clean:
                try:
                    encoded.append(encoder.transform([value])[0])
                except ValueError:
                    encoded.append(0)  # Default for unseen values
            encoded = np.array(encoded)
        
        return pd.Series(encoded, index=series.index)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns or []