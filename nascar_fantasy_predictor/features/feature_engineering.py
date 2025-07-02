"""Feature engineering for NASCAR fantasy predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3


class FeatureEngineer:
    """Engineer features for NASCAR fantasy point prediction."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        self.db = db_connection
    
    def create_features(self, target_race_date: str, 
                       lookback_races: int = 10) -> pd.DataFrame:
        """Create feature matrix for all drivers for target race."""
        # Get all active drivers
        drivers = self._get_active_drivers(target_race_date)
        
        features = []
        for driver_id in drivers:
            driver_features = self._create_driver_features(
                driver_id, target_race_date, lookback_races
            )
            features.append(driver_features)
        
        return pd.DataFrame(features)
    
    def _get_active_drivers(self, race_date: str) -> List[int]:
        """Get list of active driver IDs around the race date."""
        query = """
        SELECT DISTINCT rr.driver_id
        FROM race_results rr
        JOIN races r ON rr.race_id = r.id
        WHERE r.date >= date(?, '-90 days')
        AND r.date < ?
        ORDER BY rr.driver_id
        """
        
        cursor = self.db.execute(query, (race_date, race_date))
        return [row[0] for row in cursor.fetchall()]
    
    def _create_driver_features(self, driver_id: int, target_race_date: str,
                               lookback_races: int) -> Dict:
        """Create comprehensive feature set for a single driver."""
        features = {'driver_id': driver_id}
        
        # Get driver info
        driver_info = self._get_driver_info(driver_id)
        features.update(driver_info)
        
        # Recent performance features
        recent_results = self._get_recent_results(driver_id, target_race_date, lookback_races)
        features.update(self._calculate_performance_features(recent_results))
        
        # Track-specific features
        target_track = self._get_target_track_info(target_race_date)
        track_history = self._get_track_history(driver_id, target_track.get('track_name'))
        features.update(self._calculate_track_features(track_history, target_track))
        
        # Speed analytics features
        speed_features = self._get_speed_features(driver_id, target_race_date)
        features.update(speed_features)
        
        # Momentum and trend features
        momentum_features = self._calculate_momentum_features(recent_results)
        features.update(momentum_features)
        
        # Equipment and team features
        equipment_features = self._get_equipment_features(driver_id, target_race_date)
        features.update(equipment_features)
        
        return features
    
    def _get_driver_info(self, driver_id: int) -> Dict:
        """Get basic driver information."""
        query = "SELECT name, car_number, team, manufacturer FROM drivers WHERE id = ?"
        cursor = self.db.execute(query, (driver_id,))
        result = cursor.fetchone()
        
        if result:
            return {
                'driver_name': result[0],
                'car_number': result[1],
                'team': result[2],
                'manufacturer': result[3]
            }
        return {}
    
    def _get_recent_results(self, driver_id: int, target_date: str, 
                           num_races: int) -> List[Dict]:
        """Get recent race results for driver."""
        query = """
        SELECT rr.*, r.track_name, r.track_type, r.date
        FROM race_results rr
        JOIN races r ON rr.race_id = r.id
        WHERE rr.driver_id = ? AND r.date < ?
        ORDER BY r.date DESC
        LIMIT ?
        """
        
        cursor = self.db.execute(query, (driver_id, target_date, num_races))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _calculate_performance_features(self, results: List[Dict]) -> Dict:
        """Calculate performance-based features."""
        if not results:
            return self._get_default_performance_features()
        
        finishes = [r['finish_position'] for r in results if r['finish_position']]
        fantasy_points = [r['fantasy_points'] for r in results if r['fantasy_points']]
        
        features = {
            'avg_finish': np.mean(finishes) if finishes else 25.0,
            'median_finish': np.median(finishes) if finishes else 25.0,
            'best_finish': min(finishes) if finishes else 43,
            'worst_finish': max(finishes) if finishes else 43,
            'finish_std': np.std(finishes) if len(finishes) > 1 else 10.0,
            'avg_fantasy_points': np.mean(fantasy_points) if fantasy_points else 10.0,
            'fantasy_points_std': np.std(fantasy_points) if len(fantasy_points) > 1 else 5.0,
            'top_5_rate': sum(1 for f in finishes if f <= 5) / len(finishes) if finishes else 0.0,
            'top_10_rate': sum(1 for f in finishes if f <= 10) / len(finishes) if finishes else 0.0,
            'dnf_rate': sum(1 for r in results if r.get('dnf', False)) / len(results),
            'laps_led_avg': np.mean([r.get('laps_led', 0) for r in results]),
            'recent_races_count': len(results)
        }
        
        return features
    
    def _get_target_track_info(self, race_date: str) -> Dict:
        """Get information about the target race track."""
        query = """
        SELECT track_name, track_type, track_length, banking
        FROM races
        WHERE date = ?
        """
        
        cursor = self.db.execute(query, (race_date,))
        result = cursor.fetchone()
        
        if result:
            return {
                'track_name': result[0],
                'track_type': result[1],
                'track_length': result[2],
                'banking': result[3]
            }
        return {}
    
    def _get_track_history(self, driver_id: int, track_name: str) -> List[Dict]:
        """Get driver's historical performance at specific track."""
        if not track_name:
            return []
        
        query = """
        SELECT rr.*, r.date
        FROM race_results rr
        JOIN races r ON rr.race_id = r.id
        WHERE rr.driver_id = ? AND r.track_name = ?
        ORDER BY r.date DESC
        LIMIT 10
        """
        
        cursor = self.db.execute(query, (driver_id, track_name))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _calculate_track_features(self, track_history: List[Dict], 
                                 track_info: Dict) -> Dict:
        """Calculate track-specific features."""
        features = {
            'track_races_count': len(track_history),
            'track_avg_finish': 25.0,
            'track_best_finish': 43,
            'track_avg_fantasy_points': 10.0,
            'track_type_encoded': self._encode_track_type(track_info.get('track_type')),
            'track_length': track_info.get('track_length', 1.5),
            'track_banking': track_info.get('banking', 12.0)
        }
        
        if track_history:
            finishes = [r['finish_position'] for r in track_history if r['finish_position']]
            fantasy_points = [r['fantasy_points'] for r in track_history if r['fantasy_points']]
            
            if finishes:
                features['track_avg_finish'] = np.mean(finishes)
                features['track_best_finish'] = min(finishes)
            
            if fantasy_points:
                features['track_avg_fantasy_points'] = np.mean(fantasy_points)
        
        return features
    
    def _get_speed_features(self, driver_id: int, target_date: str) -> Dict:
        """Get speed analytics features."""
        query = """
        SELECT sa.*
        FROM speed_analytics sa
        JOIN races r ON sa.race_id = r.id
        WHERE sa.driver_id = ? AND r.date < ?
        ORDER BY r.date DESC
        LIMIT 5
        """
        
        cursor = self.db.execute(query, (driver_id, target_date))
        results = cursor.fetchall()
        
        if not results:
            return self._get_default_speed_features()
        
        green_flag_speeds = [r[3] for r in results if r[3]]  # green_flag_speed column
        late_run_speeds = [r[4] for r in results if r[4]]   # late_run_speed column
        total_speed_ratings = [r[5] for r in results if r[5]]  # total_speed_rating column
        
        return {
            'avg_green_flag_speed': np.mean(green_flag_speeds) if green_flag_speeds else 0.0,
            'avg_late_run_speed': np.mean(late_run_speeds) if late_run_speeds else 0.0,
            'avg_total_speed_rating': np.mean(total_speed_ratings) if total_speed_ratings else 0.0,
            'speed_consistency': np.std(green_flag_speeds) if len(green_flag_speeds) > 1 else 0.0
        }
    
    def _calculate_momentum_features(self, results: List[Dict]) -> Dict:
        """Calculate momentum and trend features."""
        if len(results) < 3:
            return {'momentum_score': 0.0, 'trend_direction': 0.0}
        
        # Recent 3 races vs previous 3 races
        recent_3 = results[:3]
        previous_3 = results[3:6] if len(results) >= 6 else results[3:]
        
        recent_avg = np.mean([r['fantasy_points'] for r in recent_3 if r['fantasy_points']])
        previous_avg = np.mean([r['fantasy_points'] for r in previous_3 if r['fantasy_points']])
        
        momentum_score = (recent_avg - previous_avg) if previous_avg else 0.0
        
        # Trend direction (improving = 1, declining = -1, stable = 0)
        trend_direction = 1 if momentum_score > 2 else (-1 if momentum_score < -2 else 0)
        
        return {
            'momentum_score': momentum_score,
            'trend_direction': trend_direction
        }
    
    def _get_equipment_features(self, driver_id: int, target_date: str) -> Dict:
        """Get equipment and team-related features."""
        # Get manufacturer performance
        driver_info = self._get_driver_info(driver_id)
        manufacturer = driver_info.get('manufacturer', 'Unknown')
        
        query = """
        SELECT AVG(rr.fantasy_points) as mfg_avg_points
        FROM race_results rr
        JOIN drivers d ON rr.driver_id = d.id
        JOIN races r ON rr.race_id = r.id
        WHERE d.manufacturer = ? AND r.date >= date(?, '-30 days') AND r.date < ?
        """
        
        cursor = self.db.execute(query, (manufacturer, target_date, target_date))
        result = cursor.fetchone()
        
        return {
            'manufacturer_encoded': self._encode_manufacturer(manufacturer),
            'manufacturer_avg_points': result[0] if result[0] else 15.0
        }
    
    def _encode_track_type(self, track_type: str) -> int:
        """Encode track type as integer."""
        encoding = {
            'Superspeedway': 1,
            'Intermediate': 2,
            'Short Track': 3,
            'Road Course': 4,
            'Dirt Track': 5
        }
        return encoding.get(track_type, 2)  # Default to Intermediate
    
    def _encode_manufacturer(self, manufacturer: str) -> int:
        """Encode manufacturer as integer."""
        encoding = {
            'Chevrolet': 1,
            'Ford': 2,
            'Toyota': 3
        }
        return encoding.get(manufacturer, 1)  # Default to Chevrolet
    
    def _get_default_performance_features(self) -> Dict:
        """Get default performance features for new drivers."""
        return {
            'avg_finish': 25.0,
            'median_finish': 25.0,
            'best_finish': 43,
            'worst_finish': 43,
            'finish_std': 10.0,
            'avg_fantasy_points': 10.0,
            'fantasy_points_std': 5.0,
            'top_5_rate': 0.0,
            'top_10_rate': 0.0,
            'dnf_rate': 0.1,
            'laps_led_avg': 0.0,
            'recent_races_count': 0
        }
    
    def _get_default_speed_features(self) -> Dict:
        """Get default speed features for drivers without speed data."""
        return {
            'avg_green_flag_speed': 0.0,
            'avg_late_run_speed': 0.0,
            'avg_total_speed_rating': 0.0,
            'speed_consistency': 0.0
        }