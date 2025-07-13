# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASCAR Fantasy Predictor is an AI-powered tool for making NASCAR fantasy league picks. The project is in early development with a basic CLI structure in place.

## Development Commands

### Installation and Setup
```bash
# Install the package in development mode with dev dependencies
pip install -e ".[dev]"
```

### Code Quality and Testing
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Run linting with flake8
flake8

# Type checking with mypy
mypy nascar_fantasy_predictor/

# Run tests
pytest
```

### NASCAR Predictor Usage

#### Initial Setup
```bash
# Initialize database and download 3 years of historical data
nascar-predictor init --start-year 2022

# Check system status
nascar-predictor status
```

#### Model Training
```bash
# Train model on all available historical data
nascar-predictor train model --epochs 100

# Train model with aggressive parameters for maximum accuracy
nascar-predictor train aggressive
```

#### Making Predictions
```bash
# Predict for next race with qualifying results
nascar-predictor predict race --qualifying-file qualifying_results.csv

# Predict for specific driver
nascar-predictor predict driver --driver-name "Tyler Reddick" --start-position 5

# Save predictions to file
nascar-predictor predict race --qualifying-file qualifying.csv --output predictions.csv
```

#### Model Updates
```bash
# Update model with new race data
nascar-predictor train update --since-date 2024-01-01 --model-path models/my_model

# Weekly update with 2025 data from LapRaptor.com
nascar-predictor data update-weekly --auto-retrain

# Fetch all available 2025 race data
nascar-predictor data fetch-2025

# Evaluate model accuracy on past race
nascar-predictor predict evaluate --race-date 2024-02-25 --qualifying-file qualifying.csv

# Check system status
nascar-predictor system status

# Analyze feature importance
nascar-predictor analyze features --output feature_report
```

## Project Structure

- `nascar_fantasy_predictor/` - Main package directory
  - `cli/` - Modular command line interface
    - `main.py` - Main CLI entry point
    - `commands/` - Command modules organized by functionality
      - `data_commands.py` - Data management commands
      - `training_commands.py` - Model training commands  
      - `prediction_commands.py` - Race prediction commands
      - `analysis_commands.py` - Model analysis and interpretation
      - `maintenance_commands.py` - System maintenance utilities
  - `data/` - Data collection and management
    - `database.py` - SQLite database schema and management
    - `racing_reference_scraper.py` - Racing Reference historical data scraper
    - `ifantasyrace_scraper.py` - iFantasyRace speed analytics scraper
    - `data_collector.py` - Main data collection orchestrator
  - `features/` - Feature engineering modules
    - `fantasy_points.py` - Fantasy points calculation for different scoring systems
    - `feature_engineering.py` - Comprehensive feature creation pipeline
  - `models/` - PyTorch model implementations
    - `tabular_nn.py` - Neural network architectures for tabular data
    - `trainer.py` - Model training and management system
  - `prediction/` - Prediction engine
    - `predictor.py` - Main prediction engine with uncertainty quantification
  - `interpretation/` - Model explanation and analysis
    - `feature_importance.py` - Feature importance analysis and SHAP
  - `utils/` - Utility modules
    - `logging.py` - Centralized logging configuration
    - `exceptions.py` - Custom exception classes
- `tests/` - Test directory
- `docs/` - Documentation directory
- `pyproject.toml` - Project configuration with PyTorch dependencies

## Architecture Overview

The system uses a multi-component architecture:

1. **Data Collection**: Scrapes historical race results from Racing-Reference and speed analytics from iFantasyRace
2. **Feature Engineering**: Creates 25+ features including performance metrics, track history, speed analytics, and momentum indicators
3. **PyTorch Models**: Tabular neural network architecture optimized for structured NASCAR data
4. **Prediction Engine**: Provides point predictions with confidence intervals and lineup optimization
5. **Incremental Learning**: Updates models with new race data automatically

## Key Dependencies

- **torch** - PyTorch deep learning framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **requests** - HTTP requests for data fetching
- **beautifulsoup4** - Web scraping
- **click** - Enhanced CLI interface
- **tqdm** - Progress bars
- **schedule** - Automated updates

## Data Sources

- **Racing-Reference.info**: Historical race results, driver stats, track info
- **iFantasyRace.com**: Speed analytics including green flag speed, late-run performance
- **Local SQLite Database**: Normalized schema storing races, drivers, results, and speed data

## Model Features

The feature engineering pipeline creates:
- Performance metrics (avg finish, fantasy points, consistency)
- Track-specific features (track history, track type encoding)
- Speed analytics (green flag speed, late-run speed, total speed rating)
- Momentum indicators (recent vs historical performance trends)
- Equipment factors (manufacturer performance, team stats)
- Temporal features (recent form, rolling averages)