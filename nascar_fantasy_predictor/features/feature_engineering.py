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
    
    def create_features(self, target_race_date: str, lookback_races: int = 10, driver_list: List[str] = None) -> pd.DataFrame:
        """Create feature matrix for all drivers for target race."""
        if driver_list is not None:
            # Use provided driver list (for future races)
            active_drivers = driver_list
        else:
            # Use historical active drivers (for past races)
            active_drivers = self.data_manager.get_active_drivers(target_race_date)
        
        features = []
        for driver_name in active_drivers:
            driver_features = self._create_driver_features(
                driver_name, target_race_date, lookback_races
            )
            # Only add if we got valid features (driver has historical data)
            if driver_features.get('avg_finish') is not None:
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
        
        # Start position stats (historical qualifying performance)
        start_positions = results['start_position'].dropna()
        if not start_positions.empty:
            features['avg_start'] = start_positions.mean()
            features['best_start'] = start_positions.min()
            features['qualifying_consistency'] = start_positions.std()
            features['front_row_rate'] = (start_positions <= 2).mean()
            features['top_12_start_rate'] = (start_positions <= 12).mean()
        else:
            features['avg_start'] = 25.0
            features['best_start'] = 25
            features['qualifying_consistency'] = 10.0
            features['front_row_rate'] = 0.0
            features['top_12_start_rate'] = 0.3
        
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
    
    def create_features_with_qualifying(self, target_race_date: str, 
                                       qualifying_results: pd.DataFrame = None, 
                                       lookback_races: int = 10) -> pd.DataFrame:
        """
        Create features including actual qualifying results if available.
        
        Args:
            target_race_date: Date of target race
            qualifying_results: DataFrame with driver_name and start_position columns
            lookback_races: Number of historical races to consider
            
        Returns:
            Feature DataFrame with optional qualifying data
        """
        # Get base features (historical only)
        features_df = self.create_features(target_race_date, lookback_races)
        
        # Add qualifying results if provided
        if qualifying_results is not None and not qualifying_results.empty:
            # Add current race starting position
            features_df = pd.merge(
                features_df, 
                qualifying_results[['driver_name', 'start_position']], 
                on='driver_name', 
                how='left'
            )
            
            # Rename to avoid confusion with historical avg_start
            features_df = features_df.rename(columns={'start_position': 'current_start_position'})
            
            # Ensure all drivers have qualifying results - no fallback to historical averages
            missing_qualifying = features_df['current_start_position'].isna().sum()
            if missing_qualifying > 0:
                missing_drivers = features_df[features_df['current_start_position'].isna()]['driver_name'].tolist()
                raise ValueError(f"Missing qualifying results for {missing_qualifying} drivers: {missing_drivers}. All drivers must have actual starting positions.")
            
            # Add qualifying performance vs expectation (only where historical data exists)
            features_df['start_vs_expected'] = (
                features_df['avg_start'] - features_df['current_start_position']
            )  # Positive = better than expected qualifying
            
            # Fill NaN for drivers with no historical data (new drivers)
            features_df['start_vs_expected'] = features_df['start_vs_expected'].fillna(0)
            
        return features_df
    
    def get_available_features_before_race(self) -> List[str]:
        """Get list of features that are available before race starts."""
        pre_race_features = [
            'avg_finish', 'best_finish', 'worst_finish',
            'avg_start', 'best_start', 'qualifying_consistency', 
            'front_row_rate', 'top_12_start_rate',
            'races_completed', 'dnf_rate', 'top_10_rate', 'top_5_rate',
            'track_races', 'track_avg_finish',
            'recent_avg_finish', 'form_score'
        ]
        return pre_race_features
    
    def get_race_day_features(self) -> List[str]:
        """Get list of features that require race day information."""
        race_day_features = [
            'current_start_position',  # Known after qualifying
            'start_vs_expected',       # Calculated after qualifying
            # Future race-day features could include:
            # 'weather_conditions', 'track_temperature', 'pit_strategy'
        ]
        return race_day_features

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns or []