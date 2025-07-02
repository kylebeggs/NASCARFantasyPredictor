"""Main prediction engine for NASCAR fantasy points."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

from ..data.database import DatabaseManager
from ..features.feature_engineering import FeatureEngineer
from ..features.feature_processor import FeatureProcessor
from ..features.fantasy_points import FantasyPointsCalculator
from ..models.trainer import NASCARModelTrainer


class NASCARPredictor:
    """Main prediction engine for NASCAR fantasy points."""
    
    def __init__(self, db_path: str = None, model_path: str = None):
        self.db_manager = DatabaseManager(db_path)
        self.feature_engineer = None
        self.feature_processor = FeatureProcessor()
        self.trainer = NASCARModelTrainer()
        self.fantasy_calculator = FantasyPointsCalculator()
        self.model_path = model_path
        
        # Initialize database connection
        self.db_connection = self.db_manager.get_connection()
        self.feature_engineer = FeatureEngineer(self.db_connection)
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def predict_next_race(self, race_date: str, num_predictions: int = None) -> pd.DataFrame:
        """Predict fantasy points for all drivers in the next race."""
        if self.trainer.model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Create features for all drivers
        features_df = self.feature_engineer.create_features(race_date)
        
        if features_df.empty:
            raise ValueError(f"No driver data available for race on {race_date}")
        
        # Prepare features for prediction
        feature_columns = [col for col in features_df.columns 
                          if col not in ['driver_id', 'driver_name']]
        
        X = features_df[feature_columns].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.trainer.device)
        
        # Make predictions
        predictions, uncertainties = self.trainer.predict_with_uncertainty(X_tensor)
        
        # Create results dataframe
        results = pd.DataFrame({
            'driver_id': features_df['driver_id'],
            'driver_name': features_df['driver_name'],
            'predicted_fantasy_points': predictions,
            'prediction_uncertainty': uncertainties,
            'confidence_lower': predictions - 1.96 * uncertainties,
            'confidence_upper': predictions + 1.96 * uncertainties
        })
        
        # Add current driver info
        results = self._add_driver_context(results)
        
        # Sort by predicted points (descending)
        results = results.sort_values('predicted_fantasy_points', ascending=False)
        
        # Limit results if requested
        if num_predictions:
            results = results.head(num_predictions)
        
        return results.reset_index(drop=True)
    
    def get_driver_prediction(self, driver_name: str, race_date: str) -> Dict:
        """Get detailed prediction for a specific driver."""
        # Get driver ID
        cursor = self.db_connection.execute(
            "SELECT id FROM drivers WHERE name = ?", (driver_name,)
        )
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Driver '{driver_name}' not found in database")
        
        driver_id = result[0]
        
        # Create features for this driver
        driver_features = self.feature_engineer._create_driver_features(
            driver_id, race_date, lookback_races=10
        )
        
        # Prepare for prediction
        features_df = pd.DataFrame([driver_features])
        feature_columns = [col for col in features_df.columns 
                          if col not in ['driver_id', 'driver_name']]
        
        X = features_df[feature_columns].fillna(0).values
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.trainer.device)
        
        # Make prediction
        prediction, uncertainty = self.trainer.predict_with_uncertainty(X_tensor)
        
        # Get historical context
        recent_results = self.feature_engineer._get_recent_results(
            driver_id, race_date, 5
        )
        
        return {
            'driver_name': driver_name,
            'predicted_fantasy_points': float(prediction[0]),
            'prediction_uncertainty': float(uncertainty[0]),
            'confidence_interval': [
                float(prediction[0] - 1.96 * uncertainty[0]),
                float(prediction[0] + 1.96 * uncertainty[0])
            ],
            'recent_performance': {
                'avg_fantasy_points': np.mean([r['fantasy_points'] for r in recent_results if r['fantasy_points']]),
                'avg_finish': np.mean([r['finish_position'] for r in recent_results if r['finish_position']]),
                'races_count': len(recent_results)
            },
            'feature_summary': {
                'avg_finish': driver_features.get('avg_finish'),
                'track_avg_finish': driver_features.get('track_avg_finish'),
                'momentum_score': driver_features.get('momentum_score')
            }
        }
    
    def predict_with_lineup_optimization(self, race_date: str, budget: float = 50000,
                                       salary_info: Dict[str, float] = None) -> Dict:
        """Predict with lineup optimization for DFS play."""
        predictions = self.predict_next_race(race_date)
        
        if salary_info is None:
            # Use simple salary estimation based on recent performance
            salary_info = self._estimate_salaries(predictions)
        
        # Add salary information
        predictions['salary'] = predictions['driver_name'].map(salary_info).fillna(8000)
        predictions['value_score'] = predictions['predicted_fantasy_points'] / predictions['salary'] * 1000
        
        # Optimize lineup (simple greedy approach)
        lineup = self._optimize_lineup(predictions, budget)
        
        return {
            'predictions': predictions.to_dict('records'),
            'optimal_lineup': lineup,
            'lineup_stats': {
                'total_salary': sum(driver['salary'] for driver in lineup),
                'projected_points': sum(driver['predicted_fantasy_points'] for driver in lineup),
                'remaining_budget': budget - sum(driver['salary'] for driver in lineup)
            }
        }
    
    def evaluate_predictions(self, race_date: str) -> Dict:
        """Evaluate prediction accuracy against actual results."""
        # Get actual race results
        query = """
        SELECT rr.driver_id, d.name as driver_name, rr.fantasy_points
        FROM race_results rr
        JOIN drivers d ON rr.driver_id = d.id
        JOIN races r ON rr.race_id = r.id
        WHERE r.date = ?
        """
        
        actual_results = pd.read_sql_query(query, self.db_connection, params=(race_date,))
        
        if actual_results.empty:
            raise ValueError(f"No actual results found for race on {race_date}")
        
        # Get predictions for the same race (simulate pre-race prediction)
        try:
            predictions = self.predict_next_race(race_date)
        except:
            return {"error": "Could not generate predictions for this race"}
        
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
    
    def train_model(self, start_date: str, end_date: str = None, 
                   model_type: str = "nascar", **training_kwargs):
        """Train the prediction model on historical data."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get training data
        training_data = self._get_training_data(start_date, end_date)
        
        if training_data.empty:
            raise ValueError("No training data available for the specified date range")
        
        print(f"Training on {len(training_data)} samples from {start_date} to {end_date}")
        
        # Clean and prepare data using feature processor
        cleaned_data = self.feature_processor.prepare_features_for_training(training_data, 'fantasy_points')
        
        print(f"After cleaning: {len(cleaned_data)} samples with {len(self.feature_processor.get_feature_names())} features")
        
        # Prepare data for model
        self.trainer = NASCARModelTrainer(model_type=model_type)
        X, y = self.trainer.prepare_data(cleaned_data, 'fantasy_points')
        
        # Train model
        training_result = self.trainer.train(X, y, **training_kwargs)
        
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
        X_new, y_new = self.trainer.prepare_data(new_data, 'fantasy_points')
        
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
    
    def _get_training_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Get training data from database."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get all races in date range
        race_query = """
        SELECT id, date, track_name
        FROM races
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """
        
        races = pd.read_sql_query(race_query, self.db_connection, params=(start_date, end_date))
        
        all_training_data = []
        
        for _, race in races.iterrows():
            try:
                # Create features for this race date
                features = self.feature_engineer.create_features(race['date'])
                
                # Get actual fantasy points for this race
                results_query = """
                SELECT rr.driver_id, rr.fantasy_points
                FROM race_results rr
                WHERE rr.race_id = ? AND rr.fantasy_points IS NOT NULL
                """
                
                actual_results = pd.read_sql_query(
                    results_query, self.db_connection, params=(race['id'],)
                )
                
                # Merge features with actual results
                race_data = pd.merge(features, actual_results, on='driver_id', how='inner')
                
                if not race_data.empty:
                    all_training_data.append(race_data)
                    
            except Exception as e:
                print(f"Error processing race {race['date']}: {e}")
                continue
        
        return pd.concat(all_training_data, ignore_index=True) if all_training_data else pd.DataFrame()
    
    def _add_driver_context(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add current driver context information."""
        driver_info_query = """
        SELECT id, car_number, team, manufacturer
        FROM drivers
        WHERE id IN ({})
        """.format(','.join('?' * len(predictions)))
        
        driver_info = pd.read_sql_query(
            driver_info_query, self.db_connection, 
            params=predictions['driver_id'].tolist()
        )
        
        return pd.merge(predictions, driver_info, left_on='driver_id', right_on='id', how='left')
    
    def _estimate_salaries(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Estimate DFS salaries based on recent performance."""
        # Simple salary estimation (would be replaced with actual DFS salaries)
        base_salary = 8000
        salary_range = 4000
        
        max_points = predictions['predicted_fantasy_points'].max()
        min_points = predictions['predicted_fantasy_points'].min()
        
        salaries = {}
        for _, row in predictions.iterrows():
            normalized_points = (row['predicted_fantasy_points'] - min_points) / (max_points - min_points)
            salary = base_salary + (normalized_points * salary_range)
            salaries[row['driver_name']] = int(salary / 100) * 100  # Round to nearest 100
        
        return salaries
    
    def _optimize_lineup(self, predictions: pd.DataFrame, budget: float) -> List[Dict]:
        """Simple greedy lineup optimization."""
        # Sort by value score (points per dollar)
        sorted_drivers = predictions.sort_values('value_score', ascending=False)
        
        lineup = []
        remaining_budget = budget
        
        for _, driver in sorted_drivers.iterrows():
            if driver['salary'] <= remaining_budget and len(lineup) < 6:  # Standard 6-driver lineup
                lineup.append({
                    'driver_name': driver['driver_name'],
                    'predicted_fantasy_points': driver['predicted_fantasy_points'],
                    'salary': driver['salary'],
                    'value_score': driver['value_score']
                })
                remaining_budget -= driver['salary']
        
        return lineup
    
    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()