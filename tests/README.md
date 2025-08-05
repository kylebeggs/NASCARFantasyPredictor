# NASCAR Fantasy Predictor Tests

This directory contains comprehensive tests for the NASCAR Fantasy Predictor tool.

## Test Structure

### Test Files
- `test_cli.py` - Tests for CLI commands and user interface
- `test_csv_manager.py` - Tests for data storage and retrieval functionality
- `test_feature_engineering.py` - Tests for data analysis and feature creation
- `test_fantasy_points.py` - Tests for fantasy points calculation
- `conftest.py` - Test fixtures and configuration
- `test_runner.py` - Standalone test runner script

### Test Coverage

#### CLI Commands (`test_cli.py`)
- **Basic Command Help**: Tests help text for all commands
- **Data Analysis**: Tests race analysis with latest/specific dates and track filtering
- **Driver Statistics**: Tests driver performance analysis with optional filters
- **Data Export**: Tests CSV and JSON export functionality
- **Error Handling**: Tests proper error messages for invalid inputs

#### Data Management (`test_csv_manager.py`)
- **File Operations**: Tests CSV file creation and management
- **Data Loading**: Tests race data loading with date filtering
- **Driver Queries**: Tests driver-specific data retrieval
- **Track Analysis**: Tests track-specific data filtering
- **Data Saving**: Tests saving new race results
- **Edge Cases**: Tests handling of empty data, invalid dates, missing drivers

#### Feature Engineering (`test_feature_engineering.py`)
- **Feature Creation**: Tests position change, performance flags, speed percentiles
- **Driver Statistics**: Tests summary statistics generation
- **Head-to-Head Analysis**: Tests driver comparison functionality
- **Track History**: Tests historical performance at specific tracks
- **Data Handling**: Tests NaN value handling and edge cases

#### Fantasy Points (`test_fantasy_points.py`)
- **Points Calculation**: Tests various scoring scenarios (wins, DNFs, position changes)
- **Bonus Systems**: Tests top 5, top 10, laps led bonuses
- **Scoring Systems**: Tests default and DraftKings scoring
- **Edge Cases**: Tests invalid data handling and extreme scenarios

## Running Tests

### Run All Tests
```bash
# Simple run
pytest tests/

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x

# Run specific test file
pytest tests/test_cli.py

# Run specific test
pytest tests/test_cli.py::TestCLI::test_main_help
```

### Using the Test Runner
```bash
python tests/test_runner.py
```

## Test Results

**Current Status**: ✅ 66 tests passing

### Test Categories
- **CLI Tests**: 16 tests - All passing
- **CSV Manager Tests**: 17 tests - All passing  
- **Feature Engineering Tests**: 11 tests - All passing
- **Fantasy Points Tests**: 22 tests - All passing

### Test Coverage Areas
- **Command Line Interface**: Full coverage of all commands and options
- **Data Storage & Retrieval**: Comprehensive testing of CSV operations
- **Data Analysis**: Complete coverage of feature engineering functions
- **Fantasy Scoring**: Thorough testing of points calculation logic
- **Error Handling**: Extensive edge case and error condition testing

## Adding New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` 
- Test methods: `test_<description>`

### Using Fixtures
The `conftest.py` file provides several useful fixtures:
- `sample_race_data` - Sample DataFrame with race results
- `temp_data_dir` - Temporary directory for file operations
- `csv_manager` - Clean CSV manager instance
- `csv_manager_with_data` - CSV manager pre-loaded with sample data
- `fantasy_calculator` - Fantasy points calculator instance

### Example Test
```python
def test_new_feature(csv_manager_with_data, sample_race_data):
    """Test description."""
    # Test implementation
    result = csv_manager_with_data.some_method()
    assert result is not None
```

## Continuous Integration

These tests are designed to run in CI/CD environments and provide:
- Fast execution (< 1 second total runtime)
- No external dependencies (uses temporary files)
- Clear failure messages
- Comprehensive coverage of core functionality