"""CSV-based prediction engine for NASCAR fantasy points."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..data.csv_manager import CSVDataManager
from ..data.qualifying_scraper import QualifyingScraper
from ..features.feature_engineering import FeatureEngineer
from ..features.fantasy_points import FantasyPointsCalculator
from ..models.trainer import ModelTrainer
from ..interpretation.feature_importance import FeatureImportanceAnalyzer


class NASCARPredictor:
    """Main prediction engine for NASCAR fantasy points."""
    
    def __init__(self, data_dir: str = ".", model_path: str = None):
        self.data_manager = CSVDataManager(data_dir)
        self.feature_engineer = FeatureEngineer(self.data_manager)
        self.trainer = ModelTrainer()
        self.fantasy_calculator = FantasyPointsCalculator()
        self.qualifying_scraper = QualifyingScraper()
        self.model_path = model_path
        
        # Load model if path provided
        if model_path and Path(model_path + '.pth').exists():
            self.load_model(model_path)
    
    def predict_next_race(self, race_date: str, qualifying_results: pd.DataFrame, 
                         num_predictions: int = None) -> pd.DataFrame:
        """
        Predict fantasy points for all drivers in the next race.
        
        Args:
            race_date: Date of the race
            qualifying_results: DataFrame with columns ['driver_name', 'start_position'] 
                              (REQUIRED - no predictions without qualifying)
            num_predictions: Number of top predictions to return
        
        Returns:
            DataFrame with predictions and metadata
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        if qualifying_results is None or qualifying_results.empty:
            raise ValueError("Qualifying results are required for predictions. Cannot make predictions without actual starting positions.")
        
        # For future races, manually create features for drivers with qualifying data
        driver_list = qualifying_results['driver_name'].tolist()
        
        # Get drivers who have historical data
        recent_data = self.data_manager.load_race_data()
        available_drivers = set(recent_data['driver_name'].unique())
        drivers_with_data = [d for d in driver_list if d in available_drivers]
        
        if not drivers_with_data:
            raise ValueError("No drivers in qualifying results have historical data for prediction.")
        
        # Create features manually for drivers with data
        features_list = []
        for driver_name in drivers_with_data:
            driver_features = self.feature_engineer._create_driver_features(
                driver_name, race_date, 10  # lookback races
            )
            if driver_features.get('avg_finish') is not None:
                features_list.append(driver_features)
        
        features_df = pd.DataFrame(features_list)
        
        if features_df.empty:
            raise ValueError("No valid features could be created for any drivers.")
        
        # Add qualifying data
        features_df = pd.merge(
            features_df, 
            qualifying_results[['driver_name', 'start_position']], 
            on='driver_name', 
            how='left'
        )
        
        # Rename and add derived features
        features_df = features_df.rename(columns={'start_position': 'current_start_position'})
        features_df['start_vs_expected'] = (
            features_df['avg_start'] - features_df['current_start_position']
        ).fillna(0)
        
        prediction_type = "post-qualifying"
        
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
            'confidence_upper': predictions + 1.96 * uncertainties,
            'prediction_type': prediction_type
        })
        
        # Add qualifying information if available
        if qualifying_results is not None:
            results = pd.merge(
                results,
                qualifying_results[['driver_name', 'start_position']],
                on='driver_name',
                how='left'
            )
            results = results.rename(columns={'start_position': 'qualifying_position'})
            
            # Add position change prediction
            results['predicted_position_change'] = (
                results['qualifying_position'] - results['predicted_finish_position']
            )
        
        # Add current driver info
        results = self._add_driver_context(results)
        
        # Sort by predicted finish position (ascending - lower position is better)
        results = results.sort_values('predicted_finish_position', ascending=True)
        
        # Limit results if requested
        if num_predictions:
            results = results.head(num_predictions)
        
        return results.reset_index(drop=True)
    
    def get_driver_prediction(self, driver_name: str, race_date: str, 
                             start_position: int) -> Dict:
        """
        Get detailed prediction for a specific driver.
        
        Args:
            driver_name: Name of the driver
            race_date: Date of the race
            start_position: Starting position for this race (REQUIRED)
            
        Returns:
            Dictionary with prediction details
        """
        # Check if driver exists in data
        driver_data = self.data_manager.get_driver_data(driver_name)
        if driver_data.empty:
            raise ValueError(f"Driver '{driver_name}' not found in data")
        
        # Create features for this driver
        driver_features = self.feature_engineer._create_driver_features(
            driver_name, race_date, lookback_races=10
        )
        
        # Create qualifying results DataFrame for this driver
        qualifying_results = pd.DataFrame({
            'driver_name': [driver_name],
            'start_position': [start_position]
        })
        
        # Use the enhanced feature creation with qualifying
        features_df = self.feature_engineer.create_features_with_qualifying(
            race_date, qualifying_results
        )
        
        # Filter to just this driver
        features_df = features_df[features_df['driver_name'] == driver_name]
        prediction_type = f"with start position {start_position}"
        
        if features_df.empty:
            raise ValueError(f"Could not create features for {driver_name}")
        
        if self.trainer.feature_columns is None:
            raise ValueError("Model was not trained with feature columns metadata. Retrain the model.")
        
        # Use the feature processor to ensure consistent feature preparation
        processed_df = self.feature_engineer.prepare_for_training(features_df, target_column=None)
        
        # Filter to only the features that were used during training
        available_features = [col for col in self.trainer.feature_columns if col in processed_df.columns]
        X = processed_df[available_features].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.trainer.device)
        
        # Make prediction
        prediction, uncertainty = self.trainer.predict_with_uncertainty(X_tensor)
        
        # Get historical context
        recent_results = self.data_manager.get_driver_recent_results(
            driver_name, race_date, 5
        )
        
        result = {
            'driver_name': driver_name,
            'predicted_finish_position': float(prediction[0]),
            'prediction_uncertainty': float(uncertainty[0]),
            'confidence_interval': [
                float(prediction[0] - 1.96 * uncertainty[0]),
                float(prediction[0] + 1.96 * uncertainty[0])
            ],
            'prediction_type': prediction_type,
            'recent_performance': {
                'avg_finish': recent_results['finish_position'].mean() if not recent_results.empty else None,
                'races_count': len(recent_results)
            },
            'feature_summary': {
                'avg_finish': driver_features.get('avg_finish'),
                'track_avg_finish': driver_features.get('track_avg_finish'),
                'form_score': driver_features.get('form_score')
            }
        }
        
        # Add starting position context
        result['starting_position'] = start_position
        result['predicted_position_change'] = start_position - float(prediction[0])
        result['start_vs_historical_avg'] = driver_features.get('avg_start', 25) - start_position
        
        # Provide context about the starting position impact
        if start_position <= 5:
            result['start_position_context'] = "Excellent starting position - strong finish likely"
        elif start_position <= 12:
            result['start_position_context'] = "Good starting position - solid opportunity"
        elif start_position <= 20:
            result['start_position_context'] = "Mid-pack start - potential for gains"
        else:
            result['start_position_context'] = "Back of pack start - needs strong race pace"
        
        return result
    
    def predict_with_auto_qualifying(self, race_date: str, num_predictions: int = None) -> pd.DataFrame:
        """
        Make predictions by automatically fetching qualifying results.
        
        Args:
            race_date: Date of the race
            num_predictions: Number of top predictions to return
            
        Returns:
            DataFrame with predictions and metadata
        """
        # Attempt to fetch qualifying results automatically
        try:
            qualifying_results = self.qualifying_scraper.get_qualifying_results(race_date)
            print(f"✓ Found qualifying results for {race_date}")
        except Exception as e:
            raise ValueError(f"Could not fetch qualifying results for {race_date}: {e}. Please provide qualifying results manually.")
        
        return self.predict_next_race(race_date, qualifying_results, num_predictions)
    
    def get_driver_prediction_with_auto_qualifying(self, driver_name: str, race_date: str) -> Dict:
        """
        Get driver prediction by automatically finding their starting position.
        
        Args:
            driver_name: Name of the driver
            race_date: Date of the race
            
        Returns:
            Dictionary with prediction details
        """
        # Get qualifying results
        try:
            qualifying_results = self.qualifying_scraper.get_qualifying_results(race_date)
        except Exception as e:
            raise ValueError(f"Could not fetch qualifying results for {race_date}: {e}")
        
        # Find driver's starting position
        driver_qual = qualifying_results[qualifying_results['driver_name'] == driver_name]
        if driver_qual.empty:
            available_drivers = qualifying_results['driver_name'].tolist()
            raise ValueError(f"Driver '{driver_name}' not found in qualifying results. Available drivers: {available_drivers}")
        
        start_position = int(driver_qual.iloc[0]['start_position'])
        
        return self.get_driver_prediction(driver_name, race_date, start_position)
    
    def predict_today_race(self, num_predictions: int = None) -> pd.DataFrame:
        """
        Make predictions for today's race (if there is one).
        
        Args:
            num_predictions: Number of top predictions to return
            
        Returns:
            DataFrame with predictions and metadata
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Check if today is Sonoma 2025-07-13
            if today == '2025-07-13':
                qualifying_results = self.qualifying_scraper.get_sonoma_2025_qualifying()
                print("Using Sonoma 2025 qualifying results")
                return self.predict_next_race(today, qualifying_results, num_predictions)
            else:
                return self.predict_with_auto_qualifying(today, num_predictions)
        except Exception as e:
            raise ValueError(f"No race found for today ({today}) or qualifying results unavailable: {e}")
    
    def get_qualifying_results(self, race_date: str, track_name: str = None) -> pd.DataFrame:
        """
        Get qualifying results for a specific race.
        
        Args:
            race_date: Race date in YYYY-MM-DD format
            track_name: Optional track name
            
        Returns:
            DataFrame with qualifying results
        """
        return self.qualifying_scraper.get_qualifying_results(race_date, track_name)
    
    
    def evaluate_predictions(self, race_date: str, qualifying_results: pd.DataFrame) -> Dict:
        """Evaluate prediction accuracy against actual results."""
        # Get actual race results
        actual_results = self.data_manager.get_race_results(race_date)
        
        if actual_results.empty:
            raise ValueError(f"No actual results found for race on {race_date}")
        
        if qualifying_results is None or qualifying_results.empty:
            return {"error": "Qualifying results required for evaluation"}
        
        # Get predictions for the same race using qualifying results
        try:
            predictions = self.predict_next_race(race_date, qualifying_results)
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
    
    def get_feature_importance(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comprehensive feature importance analysis from the trained model.
        
        Args:
            methods: List of methods to use ['weights', 'gradients', 'permutation', 'shap']
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get training data for importance analysis
        training_data = self._get_all_training_data()
        if training_data.empty:
            raise ValueError("No training data available for feature importance analysis.")
        
        # Prepare data using the same feature processing
        cleaned_data = self.feature_engineer.prepare_for_training(training_data, 'finish_position')
        feature_names = self.feature_engineer.get_feature_names()
        
        X = cleaned_data[feature_names].fillna(0).values
        y = cleaned_data['finish_position'].values
        
        return self.trainer.get_feature_importance(X, y, methods)
    
    def explain_driver_prediction(self, driver_name: str, race_date: str, 
                                method: str = 'gradients') -> Dict:
        """
        Explain why a specific driver received their prediction.
        
        Args:
            driver_name: Name of the driver to explain
            race_date: Race date for prediction
            method: Method to use for explanation ('gradients' or 'shap')
            
        Returns:
            Dictionary with driver prediction explanation
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get driver features
        driver_features = self.feature_engineer._create_driver_features(
            driver_name, race_date, lookback_races=10
        )
        
        # Prepare for prediction
        features_df = pd.DataFrame([driver_features])
        processed_df = self.feature_engineer.prepare_for_training(features_df, target_column=None)
        
        X = processed_df[self.trainer.feature_columns].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        
        # Get explanation
        explanation_df = self.trainer.explain_predictions(
            X_scaled, [driver_name], method
        )
        
        # Get prediction
        prediction_dict = self.get_driver_prediction(driver_name, race_date)
        
        # Get top contributing features
        feature_contributions = {}
        for col in self.trainer.feature_columns:
            if col in explanation_df.columns:
                feature_contributions[col] = float(explanation_df.iloc[0][col])
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return {
            'driver_name': driver_name,
            'race_date': race_date,
            'prediction': prediction_dict['predicted_finish_position'],
            'uncertainty': prediction_dict['prediction_uncertainty'],
            'top_positive_factors': [
                {'feature': f, 'contribution': c} 
                for f, c in sorted_features[:5] if c > 0
            ],
            'top_negative_factors': [
                {'feature': f, 'contribution': c} 
                for f, c in sorted_features[:5] if c < 0
            ],
            'all_feature_contributions': dict(sorted_features),
            'method_used': method
        }
    
    def get_global_feature_importance(self, top_k: int = 15) -> Dict[str, Any]:
        """
        Get global feature importance analysis for the model.
        
        Args:
            top_k: Number of top features to highlight
            
        Returns:
            Dictionary with global feature importance analysis
        """
        importance_df = self.get_feature_importance()
        
        # Get top features
        top_features = importance_df.head(top_k)
        
        # Create summary
        summary = {
            'total_features': len(importance_df),
            'top_features': [
                {
                    'feature': feature,
                    'importance': float(importance_df.loc[feature, 'average']),
                    'rank': i + 1
                }
                for i, feature in enumerate(top_features.index)
            ],
            'top_10_contribution': float(importance_df.head(10)['average'].sum()),
            'feature_categories': self._categorize_features(top_features.index.tolist()),
            'importance_distribution': {
                'min': float(importance_df['average'].min()),
                'max': float(importance_df['average'].max()),
                'mean': float(importance_df['average'].mean()),
                'std': float(importance_df['average'].std())
            }
        }
        
        return summary
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type for better understanding."""
        categories = {
            'performance': [],
            'track_history': [],
            'recent_form': [],
            'race_completion': [],
            'other': []
        }
        
        for feature in feature_names:
            if any(word in feature.lower() for word in ['avg_finish', 'best_finish', 'worst_finish', 'top_']):
                categories['performance'].append(feature)
            elif any(word in feature.lower() for word in ['track', 'manufacturer']):
                categories['track_history'].append(feature)
            elif any(word in feature.lower() for word in ['recent', 'form', 'momentum']):
                categories['recent_form'].append(feature)
            elif any(word in feature.lower() for word in ['dnf', 'races_completed']):
                categories['race_completion'].append(feature)
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def save_feature_importance_report(self, output_path: str, methods: Optional[List[str]] = None):
        """
        Generate and save comprehensive feature importance report.
        
        Args:
            output_path: Path to save report (without extension)
            methods: Methods to include in analysis
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get training data for importance analysis
        training_data = self._get_all_training_data()
        if training_data.empty:
            raise ValueError("No training data available for feature importance analysis.")
        
        # Prepare data
        cleaned_data = self.feature_engineer.prepare_for_training(training_data, 'finish_position')
        feature_names = self.feature_engineer.get_feature_names()
        
        X = cleaned_data[feature_names].fillna(0).values
        y = cleaned_data['finish_position'].values
        
        # Generate report
        self.trainer.save_feature_importance_report(X, y, output_path, methods)
    
    def get_track_specific_importance(self, track_name: str, 
                                    methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance analysis for a specific track.
        
        Args:
            track_name: Name of the track to analyze
            methods: Methods to use for analysis
            
        Returns:
            DataFrame with track-specific feature importance
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get all training data
        training_data = self._get_all_training_data()
        if training_data.empty:
            raise ValueError("No training data available for analysis.")
        
        # Filter for the specific track
        track_data = training_data[training_data['track_name'].str.contains(track_name, case=False, na=False)]
        if track_data.empty:
            available_tracks = training_data['track_name'].unique()
            raise ValueError(f"No data found for track '{track_name}'. Available tracks: {list(available_tracks)}")
        
        print(f"Found {len(track_data)} records for {track_name}")
        
        # Prepare features for this track's data
        cleaned_data = self.feature_engineer.prepare_for_training(track_data, 'finish_position')
        feature_names = self.feature_engineer.get_feature_names()
        
        X = cleaned_data[feature_names].fillna(0).values
        y = cleaned_data['finish_position'].values
        
        # Create track indicator (all 1s since we already filtered)
        track_indicator = np.ones(len(X))
        
        # Get track-specific importance
        analyzer = FeatureImportanceAnalyzer(
            self.trainer.model, 
            feature_names, 
            torch.device(self.trainer.device)
        )
        
        return analyzer.get_track_specific_importance(X, y, track_indicator, track_name, methods)
    
    def compare_track_importance(self, track_names: List[str], 
                               methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Compare feature importance across multiple tracks.
        
        Args:
            track_names: List of track names to compare
            methods: Methods to use for analysis
            
        Returns:
            Dictionary mapping track names to their importance DataFrames
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get all training data
        training_data = self._get_all_training_data()
        if training_data.empty:
            raise ValueError("No training data available for analysis.")
        
        track_importance = {}
        
        for track_name in track_names:
            try:
                importance_df = self.get_track_specific_importance(track_name, methods)
                track_importance[track_name] = importance_df
            except Exception as e:
                print(f"Error analyzing {track_name}: {e}")
        
        return track_importance
    
    def analyze_track_types(self, methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance by track type (superspeedway, road course, etc.).
        
        Args:
            methods: Methods to use for analysis
            
        Returns:
            Dictionary mapping track types to their importance DataFrames
        """
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Get all training data
        training_data = self._get_all_training_data()
        if training_data.empty:
            raise ValueError("No training data available for analysis.")
        
        # Classify tracks by type
        track_types = self._classify_track_types(training_data)
        
        # Prepare data for analysis
        cleaned_data = self.feature_engineer.prepare_for_training(training_data, 'finish_position')
        feature_names = self.feature_engineer.get_feature_names()
        
        X = cleaned_data[feature_names].fillna(0).values
        y = cleaned_data['finish_position'].values
        
        # Create track type indicators
        track_type_data = np.array([track_types.get(track, 0) for track in training_data['track_name']])
        
        # Get unique track types and their names
        unique_types = list(set(track_types.values()))
        type_names = {
            1: "Superspeedway",
            2: "Road Course", 
            3: "Short Track",
            4: "Intermediate",
            0: "Other"
        }
        
        track_type_names = [type_names.get(t, f"Type_{t}") for t in unique_types if t != 0]
        
        # Analyze each track type
        analyzer = FeatureImportanceAnalyzer(
            self.trainer.model, 
            feature_names, 
            torch.device(self.trainer.device)
        )
        
        return analyzer.compare_track_importance(
            X, y, track_type_data, track_type_names, 
            [t for t in unique_types if t != 0], methods
        )
    
    def _classify_track_types(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Classify tracks by type based on track name.
        
        Args:
            data: Training data with track_name column
            
        Returns:
            Dictionary mapping track names to type indicators
        """
        track_types = {}
        
        # Define track classifications
        superspeedways = ['Daytona', 'Talladega']
        road_courses = ['Sonoma', 'Watkins Glen', 'ROVAL', 'Circuit of the Americas', 'Road America', 'Indianapolis Road Course']
        short_tracks = ['Bristol', 'Martinsville', 'Richmond', 'Phoenix']
        # Everything else is intermediate
        
        for track in data['track_name'].unique():
            if pd.isna(track):
                continue
                
            track_str = str(track)
            if any(name in track_str for name in superspeedways):
                track_types[track] = 1  # Superspeedway
            elif any(name in track_str for name in road_courses):
                track_types[track] = 2  # Road Course
            elif any(name in track_str for name in short_tracks):
                track_types[track] = 3  # Short Track
            else:
                track_types[track] = 4  # Intermediate
        
        return track_types
    
    def get_sonoma_specific_insights(self) -> Dict:
        """
        Get specific insights for Sonoma Raceway (road course analysis).
        
        Returns:
            Dictionary with Sonoma-specific feature analysis
        """
        try:
            # Get Sonoma-specific feature importance
            sonoma_importance = self.get_track_specific_importance("Sonoma")
            
            # Get road course comparison
            road_courses = ["Sonoma", "Watkins Glen"]
            road_course_comparison = {}
            
            for track in road_courses:
                try:
                    road_course_comparison[track] = self.get_track_specific_importance(track)
                except:
                    continue
            
            # Analyze what makes Sonoma different
            top_features = sonoma_importance.head(5).index.tolist()
            bottom_features = sonoma_importance.tail(3).index.tolist()
            
            insights = {
                'track_name': 'Sonoma Raceway',
                'track_type': 'Road Course',
                'sample_count': getattr(sonoma_importance, 'attrs', {}).get('sample_count', 'Unknown'),
                'top_5_features': [
                    {
                        'feature': feature,
                        'importance': float(sonoma_importance.loc[feature, 'average']),
                        'rank': i + 1
                    }
                    for i, feature in enumerate(top_features)
                ],
                'least_important_features': [
                    {
                        'feature': feature,
                        'importance': float(sonoma_importance.loc[feature, 'average']),
                        'rank': len(sonoma_importance) - len(bottom_features) + i + 1
                    }
                    for i, feature in enumerate(bottom_features)
                ],
                'road_course_comparison': road_course_comparison,
                'insights': self._generate_sonoma_insights(sonoma_importance)
            }
            
            return insights
            
        except Exception as e:
            return {'error': f"Could not analyze Sonoma: {e}"}
    
    def _generate_sonoma_insights(self, importance_df: pd.DataFrame) -> List[str]:
        """Generate human-readable insights about Sonoma feature importance."""
        insights = []
        
        top_feature = importance_df.index[0]
        top_importance = importance_df.iloc[0]['average']
        
        # Analyze top feature
        if 'avg_finish' in top_feature:
            insights.append("Driver's historical average finish position is the strongest predictor at Sonoma")
        elif 'form_score' in top_feature:
            insights.append("Recent performance momentum is crucial for success at Sonoma")
        elif 'dnf_rate' in top_feature:
            insights.append("Reliability and avoiding mistakes is key at this technical road course")
        elif 'top_10_rate' in top_feature:
            insights.append("Consistency in running competitively matters most at Sonoma")
        
        # Check if track-specific features are important
        if any('track' in feature for feature in importance_df.head(3).index):
            insights.append("Track-specific experience shows strong importance at Sonoma")
        else:
            insights.append("General NASCAR skills transfer well to Sonoma - track experience less critical")
        
        # Analyze feature distribution
        top_3_importance = importance_df.head(3)['average'].sum()
        if top_3_importance > 0.6:
            insights.append("Feature importance is highly concentrated - few factors dominate predictions")
        else:
            insights.append("Feature importance is well-distributed - multiple factors contribute to success")
        
        return insights