"""Test configuration and fixtures."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nascar_fantasy_predictor.data.csv_manager import CSVDataManager
from nascar_fantasy_predictor.features.fantasy_points import FantasyPointsCalculator
from nascar_fantasy_predictor.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_race_data():
    """Sample race data for testing."""
    return pd.DataFrame(
        {
            "date": ["2024-02-25", "2024-02-25", "2024-02-25", "2024-02-25"],
            "race_name": ["Daytona 500", "Daytona 500", "Daytona 500", "Daytona 500"],
            "track_name": ["Daytona International Speedway"] * 4,
            "driver_name": [
                "Kyle Larson",
                "William Byron",
                "Christopher Bell",
                "Tyler Reddick",
            ],
            "driver_id": [5, 24, 20, 45],
            "car_number": [5, 24, 20, 45],
            "finish_position": [1, 2, 3, 4],
            "start_position": [5, 1, 8, 2],
            "laps_led": [120, 80, 20, 10],
            "total_laps": [200, 200, 200, 200],
            "team": ["Hendrick Motorsports"] * 4,
            "manufacturer": ["Chevrolet"] * 4,
            "status": ["Running"] * 4,
            "series": ["Cup Series"] * 4,
            "fantasy_points": [0.0] * 4,
            "green_flag_speed": [185.5, 184.2, 183.8, 182.9],
            "late_run_speed": [186.1, 185.0, 184.5, 183.5],
            "total_speed_rating": [92.5, 89.2, 87.1, 85.3],
            "rating": [95.2, 91.8, 88.7, 86.4],
            "green_flag_passing_diff": [8, -2, 5, -1],
            "green_flag_passes": [15, 8, 12, 6],
            "quality_passes": [12, 6, 9, 4],
            "pct_quality_passes": [80.0, 75.0, 75.0, 66.7],
            "fastest_lap": [1, 0, 0, 0],
            "top_15_laps": [150, 120, 80, 60],
            "pct_top_15_laps": [75.0, 60.0, 40.0, 30.0],
            "pct_laps_led": [60.0, 40.0, 10.0, 5.0],
            "avg_position": [3.2, 4.5, 6.8, 8.1],
            "best_position": [1, 1, 3, 2],
            "worst_position": [8, 12, 15, 18],
            "mid_position": [4, 6, 8, 10],
        }
    )


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def csv_manager(temp_data_dir):
    """CSV manager with temporary directory."""
    return CSVDataManager(data_dir=temp_data_dir)


@pytest.fixture
def csv_manager_with_data(csv_manager, sample_race_data):
    """CSV manager with sample data loaded."""
    # Save sample data to the manager
    csv_manager.main_data_file.parent.mkdir(parents=True, exist_ok=True)
    sample_race_data.to_csv(csv_manager.main_data_file, index=False)
    return csv_manager


@pytest.fixture
def fantasy_calculator():
    """Fantasy points calculator."""
    return FantasyPointsCalculator()


@pytest.fixture
def draftkings_calculator():
    """DraftKings fantasy points calculator."""
    return FantasyPointsCalculator(scoring_system="draftkings")


@pytest.fixture
def feature_engineer(csv_manager_with_data):
    """Feature engineer with sample data."""
    return FeatureEngineer(csv_manager_with_data)


@pytest.fixture
def sample_driver_results():
    """Sample results for a specific driver."""
    return {
        "finish_position": 3,
        "start_position": 8,
        "laps_led": 45,
        "fastest_laps": 1,
        "dnf": False,
        "stage1_position": 5,
        "stage2_position": 3,
    }
