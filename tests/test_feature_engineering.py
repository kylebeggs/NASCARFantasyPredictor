"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from nascar_fantasy_predictor.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering functionality."""

    def test_create_features_basic(self, feature_engineer, sample_race_data):
        """Test basic feature creation."""
        features = feature_engineer.create_features(sample_race_data)

        # Check that new columns were added
        assert "position_change" in features.columns
        assert "gained_positions" in features.columns
        assert "lost_positions" in features.columns
        assert "top5_finish" in features.columns
        assert "top10_finish" in features.columns
        assert "dnf" in features.columns

        # Check calculations
        assert features.iloc[0]["position_change"] == 4  # Kyle Larson: 5 - 1 = 4
        assert features.iloc[1]["position_change"] == -1  # William Byron: 1 - 2 = -1

        # Check boolean features
        assert features.iloc[0]["gained_positions"] == True  # Larson gained positions
        assert features.iloc[1]["gained_positions"] == False  # Byron lost positions
        assert features.iloc[0]["top5_finish"] == True  # Larson finished P1
        assert features.iloc[2]["top10_finish"] == True  # Bell finished P3

    def test_create_features_speed_percentiles(
        self, feature_engineer, sample_race_data
    ):
        """Test speed percentile calculations."""
        features = feature_engineer.create_features(sample_race_data)

        # Check that speed percentiles were calculated
        assert "green_flag_speed_percentile" in features.columns

        # Kyle Larson should have highest speed percentile (fastest)
        assert features.iloc[0]["green_flag_speed_percentile"] == 1.0
        # Tyler Reddick should have lowest (slowest)
        assert features.iloc[3]["green_flag_speed_percentile"] == 0.25

    def test_create_features_track_type(self, feature_engineer):
        """Test track type encoding."""
        data_with_track_type = pd.DataFrame(
            {
                "date": ["2024-02-25"] * 4,
                "finish_position": [1, 2, 3, 4],
                "start_position": [5, 1, 8, 2],
                "track_type": [
                    "Superspeedway",
                    "Intermediate",
                    "Short Track",
                    "Road Course",
                ],
            }
        )

        features = feature_engineer.create_features(data_with_track_type)

        assert "track_type_code" in features.columns
        assert features.iloc[0]["track_type_code"] == 1  # Superspeedway
        assert features.iloc[1]["track_type_code"] == 2  # Intermediate
        assert features.iloc[2]["track_type_code"] == 3  # Short Track
        assert features.iloc[3]["track_type_code"] == 4  # Road Course

    def test_get_driver_summary_stats(self, feature_engineer):
        """Test driver summary statistics."""
        stats = feature_engineer.get_driver_summary_stats(
            "Kyle Larson", lookback_races=10
        )

        assert isinstance(stats, dict)
        assert "driver_name" in stats
        assert "races_analyzed" in stats
        assert "avg_finish" in stats
        assert "best_finish" in stats
        assert "worst_finish" in stats
        assert "top5_rate" in stats
        assert "top10_rate" in stats
        assert "consistency" in stats

        assert stats["driver_name"] == "Kyle Larson"
        assert stats["races_analyzed"] > 0
        assert stats["best_finish"] == 1  # Kyle Larson won in our sample data

    def test_get_driver_summary_stats_empty(self, feature_engineer):
        """Test driver summary stats for non-existent driver."""
        stats = feature_engineer.get_driver_summary_stats("Nonexistent Driver")

        assert stats == {}

    def test_get_matchup_analysis(self, feature_engineer):
        """Test head-to-head matchup analysis."""
        matchup = feature_engineer.get_matchup_analysis(
            "Kyle Larson", "William Byron", lookback_races=5
        )

        assert isinstance(matchup, pd.DataFrame)
        if not matchup.empty:
            expected_columns = [
                "date",
                "track_name",
                "Kyle Larson_finish",
                "William Byron_finish",
                "Kyle Larson_won",
            ]
            for col in expected_columns:
                assert col in matchup.columns

    def test_get_track_history(self, feature_engineer):
        """Test track history retrieval."""
        history = feature_engineer.get_track_history(
            "Daytona International Speedway", num_races=3
        )

        assert isinstance(history, pd.DataFrame)
        if not history.empty:
            assert all("Daytona" in track for track in history["track_name"])


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""

    def test_create_features_missing_columns(self, feature_engineer):
        """Test feature creation with missing columns."""
        minimal_data = pd.DataFrame(
            {"date": ["2024-02-25"], "driver_name": ["Test Driver"]}
        )

        # Should not raise an error
        features = feature_engineer.create_features(minimal_data)

        # Should still have original columns
        assert "date" in features.columns
        assert "driver_name" in features.columns

    def test_create_features_empty_dataframe(self, feature_engineer):
        """Test feature creation with empty dataframe."""
        empty_df = pd.DataFrame()

        features = feature_engineer.create_features(empty_df)

        assert features.empty

    def test_get_driver_summary_stats_with_nans(self, csv_manager):
        """Test driver stats with NaN values in data."""
        # Create data with some NaN values
        data_with_nans = pd.DataFrame(
            {
                "date": ["2024-02-25", "2024-03-01"],
                "driver_name": ["Test Driver", "Test Driver"],
                "finish_position": [1, np.nan],
                "start_position": [5, 3],
                "laps_led": [50, np.nan],
            }
        )

        # Save to CSV manager
        data_with_nans.to_csv(csv_manager.main_data_file, index=False)

        engineer = FeatureEngineer(csv_manager)
        stats = engineer.get_driver_summary_stats("Test Driver")

        # Should handle NaN values gracefully
        assert isinstance(stats, dict)
        assert "avg_finish" in stats
        # avg_finish should be calculated ignoring NaN
        assert not pd.isna(stats["avg_finish"])

    def test_track_type_unknown_values(self, feature_engineer):
        """Test track type encoding with unknown values."""
        data_with_unknown_track = pd.DataFrame(
            {
                "date": ["2024-02-25"],
                "track_type": ["Unknown Track Type"],
                "finish_position": [1],
            }
        )

        features = feature_engineer.create_features(data_with_unknown_track)

        assert "track_type_code" in features.columns
        assert features.iloc[0]["track_type_code"] == 0  # Unknown should map to 0

    def test_speed_percentiles_single_race(self, feature_engineer):
        """Test speed percentile calculation with single driver."""
        single_driver_data = pd.DataFrame(
            {
                "date": ["2024-02-25"],
                "green_flag_speed": [185.5],
                "finish_position": [1],
            }
        )

        features = feature_engineer.create_features(single_driver_data)

        # Single driver should get 100th percentile
        assert features.iloc[0]["green_flag_speed_percentile"] == 1.0
