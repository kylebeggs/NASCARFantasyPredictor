# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASCAR Fantasy Predictor is a data collection and analysis tool for NASCAR fantasy leagues. It focuses on gathering historical race data and providing insights to help humans make informed fantasy picks.

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

### NASCAR Data Tool Usage

#### Data Collection
```bash
# Collect historical NASCAR data (default: 2022 to current year)
nascar-predictor collect

# Collect specific years
nascar-predictor collect --start-year 2020 --end-year 2023
```

#### Data Analysis
```bash
# Analyze latest race
nascar-predictor analyze --date latest

# Analyze specific race
nascar-predictor analyze --date 2024-02-25

# Analyze track history
nascar-predictor analyze --track "Daytona"
```

#### Driver Statistics
```bash
# Get driver statistics
nascar-predictor driver-stats --driver "Kyle Larson"

# Get driver stats for specific track
nascar-predictor driver-stats --driver "Kyle Larson" --track "Phoenix"

# Get driver stats for specific year
nascar-predictor driver-stats --driver "Kyle Larson" --year 2024
```

#### Data Export
```bash
# Export all data to CSV
nascar-predictor export --output all_races.csv

# Export specific year
nascar-predictor export --output 2024_races.csv --year 2024

# Export as JSON
nascar-predictor export --output races.json --format json
```

## Project Structure

- `nascar_fantasy_predictor/` - Main package directory
  - `cli/` - Simple command line interface
    - `main.py` - CLI entry point with focused commands
  - `data/` - Data collection and management
    - `csv_manager.py` - CSV-based data storage and retrieval
    - `lapraptor_scraper.py` - LapRaptor.com speed analytics scraper
    - `nascar_official_scraper.py` - NASCAR official site scraper
    - `qualifying_scraper.py` - Qualifying results scraper
  - `features/` - Feature engineering for analysis
    - `fantasy_points.py` - Fantasy points calculation for different scoring systems
    - `feature_engineering.py` - Data analysis feature creation
  - `utils/` - Utility modules
    - `logging.py` - Centralized logging configuration
    - `exceptions.py` - Custom exception classes
- `tests/` - Test directory
- `docs/` - Documentation directory
- `pyproject.toml` - Project configuration

## Architecture Overview

The system is designed for simplicity and data analysis:

1. **Data Collection**: Scrapes historical race results and speed analytics from multiple sources
2. **Data Storage**: Simple CSV-based storage for easy access and manipulation
3. **Feature Engineering**: Creates analytical features for human interpretation
4. **Analysis Tools**: Provides insights on driver performance, track history, and head-to-head matchups
5. **Export Functionality**: Easy export to CSV/JSON for external analysis tools

## Key Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **requests** - HTTP requests for data fetching
- **beautifulsoup4** - Web scraping
- **click** - Command line interface
- **tqdm** - Progress bars

## Data Sources

- **LapRaptor.com**: Speed analytics including green flag speed, late-run performance
- **NASCAR Official Site**: Official race results and statistics
- **Local CSV Storage**: Simple file-based storage for collected data

## Analysis Features

The tool provides:
- Driver performance statistics (average finish, consistency, track-specific performance)
- Head-to-head driver comparisons
- Track history analysis
- Position change analysis
- Fantasy points calculation (DraftKings, FanDuel)
- Speed percentile rankings
- Performance trends over time

## Usage Philosophy

This tool is designed to augment human decision-making, not replace it. It provides data and insights that help identify:
- Drivers with strong recent form
- Track-specific specialists
- Consistent performers vs high-variance drivers
- Value plays based on starting position
- Historical performance patterns

The human user combines these insights with additional factors like:
- Weather conditions
- Team changes
- Equipment updates
- Driver motivations
- Current point standings

This combination of data-driven insights and human judgment leads to better fantasy predictions than either approach alone.