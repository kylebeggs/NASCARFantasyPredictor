"""Tests for CSV data manager."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nascar_fantasy_predictor.data.csv_manager import CSVDataManager


class TestCSVDataManager:
    """Test CSV data manager functionality."""

    def test_initialization(self, temp_data_dir):
        """Test CSV manager initialization."""
        manager = CSVDataManager(data_dir=temp_data_dir)

        assert manager.data_dir == Path(temp_data_dir)
        assert manager.main_data_file.exists()
        assert manager.model_history_file.exists()

    def test_empty_files_creation(self, temp_data_dir):
        """Test that empty CSV files are created with correct structure."""
        manager = CSVDataManager(data_dir=temp_data_dir)

        # Check main data file
        df = pd.read_csv(manager.main_data_file)
        expected_columns = [
            "date",
            "race_name",
            "track_name",
            "driver_name",
            "driver_id",
            "car_number",
            "finish_position",
            "start_position",
            "laps_led",
            "total_laps",
            "team",
            "manufacturer",
            "status",
            "series",
            "fantasy_points",
            "green_flag_speed",
            "late_run_speed",
        ]
        for col in expected_columns:
            assert col in df.columns
        assert len(df) == 0

    def test_load_race_data(self, csv_manager_with_data):
        """Test loading race data."""
        data = csv_manager_with_data.load_race_data()

        assert not data.empty
        assert len(data) == 4
        assert "Kyle Larson" in data["driver_name"].values
        assert "William Byron" in data["driver_name"].values

    def test_load_race_data_with_date_filter(self, csv_manager_with_data):
        """Test loading race data with date filtering."""
        data = csv_manager_with_data.load_race_data(
            start_date="2024-02-25", end_date="2024-02-25"
        )

        assert not data.empty
        assert all(data["date"] == "2024-02-25")

    def test_get_driver_data(self, csv_manager_with_data):
        """Test getting data for specific driver."""
        data = csv_manager_with_data.get_driver_data(driver_name="Kyle Larson")

        assert not data.empty
        assert all(data["driver_name"] == "Kyle Larson")
        assert len(data) == 1

    def test_get_driver_data_by_id(self, csv_manager_with_data):
        """Test getting data for driver by ID."""
        data = csv_manager_with_data.get_driver_data(driver_id=5)

        assert not data.empty
        assert all(data["driver_id"] == 5)
        assert data.iloc[0]["driver_name"] == "Kyle Larson"

    def test_get_track_data(self, csv_manager_with_data):
        """Test getting data for specific track."""
        data = csv_manager_with_data.get_track_data("Daytona International Speedway")

        assert not data.empty
        assert all("Daytona" in track for track in data["track_name"])
        assert len(data) == 4

    def test_get_race_results(self, csv_manager_with_data):
        """Test getting results for specific race."""
        data = csv_manager_with_data.get_race_results("2024-02-25")

        assert not data.empty
        assert all(data["date"] == "2024-02-25")
        assert len(data) == 4

    def test_get_active_drivers(self, csv_manager_with_data):
        """Test getting active drivers list."""
        drivers = csv_manager_with_data.get_active_drivers("2024-02-26")

        assert isinstance(drivers, list)
        assert "Kyle Larson" in drivers
        assert "William Byron" in drivers
        assert len(drivers) == 4

    def test_get_driver_recent_results(self, csv_manager_with_data):
        """Test getting driver's recent results."""
        results = csv_manager_with_data.get_driver_recent_results(
            "Kyle Larson", "2024-02-26", num_races=5
        )

        assert not results.empty
        assert all(results["driver_name"] == "Kyle Larson")
        assert len(results) == 1

    def test_get_driver_track_history(self, csv_manager_with_data):
        """Test getting driver's track history."""
        history = csv_manager_with_data.get_driver_track_history(
            "Kyle Larson", "Daytona International Speedway"
        )

        assert not history.empty
        assert all(history["driver_name"] == "Kyle Larson")
        assert all("Daytona" in track for track in history["track_name"])

    def test_save_race_results(self, csv_manager):
        """Test saving race results."""
        # Create sample data
        results = pd.DataFrame(
            {
                "driver_name": ["Driver A", "Driver B"],
                "finish_position": [1, 2],
                "start_position": [3, 1],
                "laps_led": [50, 20],
            }
        )

        csv_manager.save_race_results(results, "2024-03-01", "Test Race")

        # Verify data was saved
        saved_data = csv_manager.load_race_data()
        assert not saved_data.empty
        assert "Driver A" in saved_data["driver_name"].values
        assert "Test Race" in saved_data["race_name"].values

    def test_get_data_stats(self, csv_manager_with_data):
        """Test getting data statistics."""
        stats = csv_manager_with_data.get_data_stats()

        assert isinstance(stats, dict)
        assert "total_races" in stats
        assert "drivers" in stats
        assert "date_range" in stats
        assert stats["total_races"] > 0
        assert stats["drivers"] > 0


class TestCSVDataManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_load_empty_data(self, csv_manager):
        """Test loading data when no data exists."""
        data = csv_manager.load_race_data()
        assert data.empty

    def test_get_nonexistent_driver(self, csv_manager_with_data):
        """Test getting data for non-existent driver."""
        data = csv_manager_with_data.get_driver_data(driver_name="Nonexistent Driver")
        assert data.empty

    def test_get_nonexistent_track(self, csv_manager_with_data):
        """Test getting data for non-existent track."""
        data = csv_manager_with_data.get_track_data("Nonexistent Track")
        assert data.empty

    def test_get_results_invalid_date(self, csv_manager_with_data):
        """Test getting results for invalid date."""
        data = csv_manager_with_data.get_race_results("2020-01-01")
        assert data.empty

    def test_active_drivers_no_recent_data(self, csv_manager_with_data):
        """Test getting active drivers when no recent data exists."""
        drivers = csv_manager_with_data.get_active_drivers("2020-01-01")
        assert isinstance(drivers, list)
        assert len(drivers) == 0

    def test_save_empty_results(self, csv_manager):
        """Test saving empty race results."""
        empty_results = pd.DataFrame()

        # Should not raise an error
        csv_manager.save_race_results(empty_results, "2024-03-01", "Empty Race")

        # Data should still be empty
        data = csv_manager.load_race_data()
        assert data.empty
