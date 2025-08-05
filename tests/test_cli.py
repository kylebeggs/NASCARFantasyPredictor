"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from nascar_fantasy_predictor.cli.main import main


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def test_data_file(self, sample_race_data):
        """Create a temporary data file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_race_data.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "NASCAR Data Collection and Analysis Tool" in result.output
        assert "analyze" in result.output
        assert "collect" in result.output
        assert "driver-stats" in result.output
        assert "export" in result.output

    def test_collect_help(self, runner):
        """Test collect command help."""
        result = runner.invoke(main, ["collect", "--help"])
        assert result.exit_code == 0
        assert "Collect NASCAR race data" in result.output
        assert "--year" in result.output

    def test_collect_unsupported_year(self, runner):
        """Test collect command with unsupported year."""
        result = runner.invoke(main, ["collect", "--year", "2023"])
        assert result.exit_code == 0
        assert "Currently only 2025 data collection is supported" in result.output

    def test_analyze_help(self, runner):
        """Test analyze command help."""
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze race data and generate insights" in result.output
        assert "--date" in result.output
        assert "--track" in result.output

    def test_analyze_no_parameters(self, runner):
        """Test analyze command without parameters."""
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code == 0
        assert "Please specify --date or --track" in result.output

    def test_driver_stats_help(self, runner):
        """Test driver-stats command help."""
        result = runner.invoke(main, ["driver-stats", "--help"])
        assert result.exit_code == 0
        assert "Get detailed statistics for a specific driver" in result.output
        assert "--driver" in result.output

    def test_driver_stats_missing_driver(self, runner):
        """Test driver-stats command without driver parameter."""
        result = runner.invoke(main, ["driver-stats"])
        assert result.exit_code != 0  # Should fail due to required parameter

    def test_export_help(self, runner):
        """Test export command help."""
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "Export collected data for external analysis" in result.output
        assert "--format" in result.output
        assert "--output" in result.output
        assert "--year" in result.output


class TestCLIWithData:
    """Test CLI commands with actual data."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_analyze_with_data(self, runner, csv_manager_with_data):
        """Test analyze command with real data."""
        with runner.isolated_filesystem():
            # Copy the test data to the current directory
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            # Test analyze latest
            result = runner.invoke(main, ["analyze", "--date", "latest"])
            assert result.exit_code == 0
            assert "Race Analysis" in result.output
            assert "Top 10 Fantasy Performers" in result.output
            assert "Biggest Movers" in result.output
            assert "Kyle Larson" in result.output

    def test_analyze_specific_date(self, runner, csv_manager_with_data):
        """Test analyze command with specific date."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(main, ["analyze", "--date", "2024-02-25"])
            assert result.exit_code == 0
            assert "Race Analysis for 2024-02-25" in result.output

    def test_analyze_by_track(self, runner, csv_manager_with_data):
        """Test analyze command by track."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(main, ["analyze", "--track", "Daytona"])
            assert result.exit_code == 0
            assert "Race Analysis for Daytona" in result.output

    def test_driver_stats_with_data(self, runner, csv_manager_with_data):
        """Test driver-stats command with real data."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(main, ["driver-stats", "--driver", "Kyle Larson"])
            assert result.exit_code == 0
            assert "Driver Statistics: Kyle Larson" in result.output
            assert "Races:" in result.output
            assert "Average Finish:" in result.output
            assert "Best Finish:" in result.output
            assert "Top 5s:" in result.output

    def test_driver_stats_with_track_filter(self, runner, csv_manager_with_data):
        """Test driver-stats command with track filter."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(
                main, ["driver-stats", "--driver", "Kyle Larson", "--track", "Daytona"]
            )
            assert result.exit_code == 0
            assert "Driver Statistics: Kyle Larson" in result.output

    def test_export_csv(self, runner, csv_manager_with_data):
        """Test export command to CSV."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(main, ["export", "--output", "test_export.csv"])
            assert result.exit_code == 0
            assert "Exported" in result.output
            assert "race results to test_export.csv" in result.output

            # Check that file was created
            assert Path("test_export.csv").exists()

            # Check file contents
            df = pd.read_csv("test_export.csv")
            assert len(df) > 0
            assert "driver_name" in df.columns
            assert "finish_position" in df.columns
            assert "position_change" in df.columns  # Feature engineering column

    def test_export_json(self, runner, csv_manager_with_data):
        """Test export command to JSON."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(
                main, ["export", "--output", "test_export.json", "--format", "json"]
            )
            assert result.exit_code == 0
            assert "Exported" in result.output
            assert "race results to test_export.json" in result.output

            # Check that file was created
            assert Path("test_export.json").exists()

            # Check file contents
            with open("test_export.json", "r") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "driver_name" in data[0]

    def test_export_specific_year(self, runner, csv_manager_with_data):
        """Test export command for specific year."""
        with runner.isolated_filesystem():
            import shutil

            shutil.copy(csv_manager_with_data.main_data_file, "nascar_master_data.csv")

            result = runner.invoke(
                main, ["export", "--output", "test_2024.csv", "--year", "2024"]
            )
            assert result.exit_code == 0
            assert "Exported" in result.output

            # Check that only 2024 data is exported
            df = pd.read_csv("test_2024.csv")
            dates = pd.to_datetime(df["date"])
            assert all(dates.dt.year == 2024)
