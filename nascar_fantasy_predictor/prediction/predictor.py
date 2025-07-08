"""CSV-based prediction engine for NASCAR fantasy points."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ..data.csv_manager import CSVDataManager
from ..features.feature_engineering import FeatureEngineer
from ..features.fantasy_points import FantasyPointsCalculator
from ..models.trainer import ModelTrainer


class NASCARPredictor:
    """Main prediction engine for NASCAR fantasy points."""
    
    def __init__(self, data_dir: str = ".", model_path: str = None):
        self.data_manager = CSVDataManager(data_dir)
        self.feature_engineer = FeatureEngineer(self.data_manager)
        self.trainer = ModelTrainer()
        self.fantasy_calculator = FantasyPointsCalculator()
        self.model_path = model_path
        
        # Load model if path provided
        if model_path and Path(model_path + '.pth').exists():
            self.load_model(model_path)
    
    def predict_next_race(self, race_date: str, num_predictions: int = None) -> pd.DataFrame:
        """Predict fantasy points for all drivers in the next race."""
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Create features for all drivers
        features_df = self.feature_engineer.create_features(race_date)
        
        if features_df.empty:
            raise ValueError(f"No driver data available for race on {race_date}")
        
        # Prepare features for prediction using the same logic as training
        if self.trainer.feature_columns is None:
            raise ValueError("Model was not trained with feature columns metadata. Retrain the model.")
        
        # Use the feature processor to ensure consistent feature preparation
        processed_df = self.feature_engineer.prepare_for_training(features_df, target_column=None)
        
        # Filter to only the features that were used during training
        X = processed_df[self.trainer.feature_columns].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.trainer.device)
        
        # Make predictions
        predictions, uncertainties = self.trainer.predict_with_uncertainty(X_tensor)
        
        # Create results dataframe
        results = pd.DataFrame({
            'driver_name': features_df['driver_name'],
            'predicted_finish_position': predictions,
            'prediction_uncertainty': uncertainties,
            'confidence_lower': predictions - 1.96 * uncertainties,
            'confidence_upper': predictions + 1.96 * uncertainties
        })
        
        # Add current driver info
        results = self._add_driver_context(results)
        
        # Sort by predicted finish position (ascending - lower position is better)
        results = results.sort_values('predicted_finish_position', ascending=True)
        
        # Limit results if requested
        if num_predictions:
            results = results.head(num_predictions)
        
        return results.reset_index(drop=True)
    
    def get_driver_prediction(self, driver_name: str, race_date: str) -> Dict:
        """Get detailed prediction for a specific driver."""
        # Check if driver exists in data
        driver_data = self.data_manager.get_driver_data(driver_name)
        if driver_data.empty:
            raise ValueError(f"Driver '{driver_name}' not found in data")
        
        # Create features for this driver
        driver_features = self.feature_engineer._create_driver_features(
            driver_name, race_date, lookback_races=10
        )
        
        # Prepare for prediction using the same logic as training
        features_df = pd.DataFrame([driver_features])
        
        if self.trainer.feature_columns is None:
            raise ValueError("Model was not trained with feature columns metadata. Retrain the model.")
        
        # Use the feature processor to ensure consistent feature preparation
        processed_df = self.feature_engineer.prepare_for_training(features_df, target_column=None)
        
        # Filter to only the features that were used during training
        X = processed_df[self.trainer.feature_columns].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.trainer.device)
        
        # Make prediction
        prediction, uncertainty = self.trainer.predict_with_uncertainty(X_tensor)
        
        # Get historical context
        recent_results = self.data_manager.get_driver_recent_results(
            driver_name, race_date, 5
        )
        
        return {
            'driver_name': driver_name,
            'predicted_finish_position': float(prediction[0]),
            'prediction_uncertainty': float(uncertainty[0]),
            'confidence_interval': [
                float(prediction[0] - 1.96 * uncertainty[0]),
                float(prediction[0] + 1.96 * uncertainty[0])
            ],
            'recent_performance': {
                'avg_finish': recent_results['finish_position'].mean() if not recent_results.empty else None,
                'races_count': len(recent_results)
            },
            'feature_summary': {
                'avg_finish': driver_features.get('avg_finish'),
                'track_avg_finish': driver_features.get('track_avg_finish'),
                'momentum_score': driver_features.get('momentum_score')
            }
        }
    
    
    def evaluate_predictions(self, race_date: str) -> Dict:
        """Evaluate prediction accuracy against actual results."""
        # Get actual race results
        actual_results = self.data_manager.get_race_results(race_date)
        
        if actual_results.empty:
            raise ValueError(f"No actual results found for race on {race_date}")
        
        # Get predictions for the same race (simulate pre-race prediction)
        try:
            predictions = self.predict_next_race(race_date)
        except:
            return {"error": "Could not generate predictions for this race"}
        
        # Convert predicted finish position to fantasy points for evaluation
        # Simple conversion: better finish = more fantasy points
        predictions['predicted_fantasy_points'] = 43 - predictions['predicted_finish_position']
        
        # Calculate actual fantasy points if not present
        if 'fantasy_points' not in actual_results.columns:
            actual_results['fantasy_points'] = 43 - actual_results['finish_position']
        
        # Merge predictions with actual results
        merged = pd.merge(
            predictions[['driver_name', 'predicted_fantasy_points']],
            actual_results[['driver_name', 'fantasy_points']],
            on='driver_name',
            how='inner'
        )
        
        if merged.empty:
            return {"error": "No matching drivers between predictions and actual results"}
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(merged['predicted_fantasy_points'] - merged['fantasy_points']))
        rmse = np.sqrt(np.mean((merged['predicted_fantasy_points'] - merged['fantasy_points']) ** 2))
        correlation = merged['predicted_fantasy_points'].corr(merged['fantasy_points'])
        
        # Top 10 accuracy
        actual_top10 = set(merged.nlargest(10, 'fantasy_points')['driver_name'])
        predicted_top10 = set(merged.nlargest(10, 'predicted_fantasy_points')['driver_name'])
        top10_overlap = len(actual_top10.intersection(predicted_top10))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'top10_accuracy': top10_overlap / 10,
            'drivers_evaluated': len(merged),
            'best_prediction': merged.loc[merged['predicted_fantasy_points'].idxmax()]['driver_name'],
            'worst_prediction': merged.loc[merged['predicted_fantasy_points'].idxmin()]['driver_name']
        }
    
    def train_model(self, **training_kwargs):
        """Train the prediction model on all available historical data."""
        # Get all available training data
        training_data = self._get_all_training_data()
        
        if training_data.empty:
            raise ValueError("No training data available")
        
        # Get data range for logging
        data_stats = self.data_manager.get_data_stats()
        date_range = data_stats.get('date_range', 'unknown')
        
        if isinstance(date_range, dict):
            start_date = date_range.get('earliest', 'unknown')
            end_date = date_range.get('latest', 'unknown') 
            date_range_str = f"{start_date} to {end_date}"
        else:
            date_range_str = str(date_range)
        
        print(f"Training on {len(training_data)} samples from {date_range_str}")
        
        # Clean and prepare data using feature processor
        cleaned_data = self.feature_engineer.prepare_for_training(training_data, 'finish_position')
        
        print(f"After cleaning: {len(cleaned_data)} samples with {len(self.feature_engineer.get_feature_names())} features")
        
        # Prepare data for model
        self.trainer = ModelTrainer()
        
        # Use the feature processor's feature names for consistency
        feature_names = self.feature_engineer.get_feature_names()
        X, y = self.trainer.prepare_data(cleaned_data, target_column='finish_position', feature_columns=feature_names)
        
        # Train model
        training_result = self.trainer.train(X, y, **training_kwargs)
        
        # Log training session
        if isinstance(date_range, dict):
            start_date_log = date_range.get('earliest', 'unknown')
            end_date_log = date_range.get('latest', 'unknown')
        else:
            start_date_log = date_range_str.split(' to ')[0] if ' to ' in date_range_str else 'unknown'
            end_date_log = date_range_str.split(' to ')[1] if ' to ' in date_range_str else 'unknown'
            
        self.data_manager.log_model_training(
            model_version=f"tabular_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            train_data_start=start_date_log,
            train_data_end=end_date_log,
            model_accuracy=training_result['val_mae'],
            model_path=self.model_path or "auto_saved",
            epochs_trained=training_result['epochs_trained'],
            train_mae=training_result['train_mae'],
            val_mae=training_result['val_mae']
        )
        
        print(f"Training completed. Final validation MAE: {training_result['val_mae']:.2f}")
        
        return training_result
    
    def update_model(self, since_date: str):
        """Incrementally update model with new race data."""
        if self.trainer.model is None:
            raise ValueError("No trained model to update. Train a model first.")
        
        # Get new training data
        new_data = self._get_training_data(since_date)
        
        if new_data.empty:
            print("No new data available for model update")
            return
        
        print(f"Updating model with {len(new_data)} new samples since {since_date}")
        
        # Prepare new data
        X_new, y_new = self.trainer.prepare_data(new_data, 'finish_position')
        
        # Incremental training
        self.trainer.incremental_train(X_new, y_new)
        
        print("Model update completed")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.trainer.model is None:
            raise ValueError("No trained model to save")
        
        self.trainer.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.trainer.load_model(filepath)
        self.model_path = filepath
        print(f"Model loaded from {filepath}")
    
    def _get_all_training_data(self) -> pd.DataFrame:
        """Get all available training data from CSV files."""
        # Get all available race data
        all_data = self.data_manager.load_race_data()
        
        if all_data.empty:
            return pd.DataFrame()
        
        # Get date range
        all_data['date'] = pd.to_datetime(all_data['date'])
        min_date = all_data['date'].min().strftime('%Y-%m-%d')
        max_date = all_data['date'].max().strftime('%Y-%m-%d')
        
        return self._get_training_data(min_date, max_date)
    
    def _get_training_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Get training data from CSV files."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get all race data in date range
        race_data = self.data_manager.get_training_data(start_date, end_date)
        
        if race_data.empty:
            return pd.DataFrame()
        
        # Get unique race dates
        race_dates = sorted(race_data['date'].unique())
        
        all_training_data = []
        
        for race_date in race_dates:
            try:
                # Create features for this race date
                features = self.feature_engineer.create_features(race_date)
                
                # Get actual results for this race
                actual_results = race_data[race_data['date'] == race_date]
                
                # Select columns that exist
                result_columns = ['driver_name', 'finish_position']
                if 'fantasy_points' in actual_results.columns:
                    result_columns.append('fantasy_points')
                
                # Merge features with actual results
                merged_data = pd.merge(
                    features, 
                    actual_results[result_columns], 
                    on='driver_name', 
                    how='inner'
                )
                
                if not merged_data.empty:
                    all_training_data.append(merged_data)
                    
            except Exception as e:
                print(f"Error processing race {race_date}: {e}")
                continue
        
        return pd.concat(all_training_data, ignore_index=True) if all_training_data else pd.DataFrame()
    
    def _add_driver_context(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add current driver context information."""
        # Get driver info from recent data
        all_data = self.data_manager.load_race_data()
        
        if all_data.empty:
            return predictions
        
        # Get most recent info for each driver
        driver_info = all_data.groupby('driver_name').last().reset_index()
        driver_info = driver_info[['driver_name', 'car_number', 'team', 'manufacturer']]
        
        return pd.merge(predictions, driver_info, on='driver_name', how='left')
    
    
    def get_data_stats(self) -> Dict:
        """Get statistics about the CSV data."""
        return self.data_manager.get_data_stats()
    
    def migrate_existing_data(self, nascar_file: str = None) -> int:
        """Migrate existing CSV files to the master data format."""
        return self.data_manager.migrate_existing_csvs(nascar_file)