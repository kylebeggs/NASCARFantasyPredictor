# NASCAR Fantasy Predictor

> **Note:** This repository is 99% written by Claude (Anthropic's AI assistant) and represents a collaborative pet project combining my passion for NASCAR with exploring the capabilities of modern AI coding tools. It demonstrates how AI can rapidly prototype, implement, and iterate on complex data science projects when guided by domain expertise.

## Overview

NASCAR Fantasy Predictor is an AI-powered tool for making NASCAR fantasy league picks using machine learning models trained on historical race data and real-time performance analytics.

## Features

- **Historical Data Collection**: Scrapes race results from multiple sources including Racing-Reference and LapRaptor.com
- **Advanced Feature Engineering**: Creates 43+ features including performance metrics, track history, speed analytics, and momentum indicators
- **PyTorch Neural Networks**: Tabular deep learning models optimized for NASCAR race prediction
- **Real-time Updates**: Weekly data updates with automatic model retraining
- **Track-Specific Predictions**: Specialized predictions for different track types (road courses, short tracks, superspeedways)
- **Fantasy Optimization**: Finish position predictions with confidence intervals for fantasy league strategy

## Quick Start

### Installation
```bash
# Install the package in development mode
pip install -e ".[dev]"
```

### Initial Setup
```bash
# Initialize database and download historical data
nascar-predictor init --start-year 2022

# Check system status
nascar-predictor status
```

### Train Model
```bash
# Train model on all available historical data
nascar-predictor train
```

### Make Predictions
```bash
# Predict for next race (auto-detects next Sunday)
nascar-predictor predict

# Predict for specific race date
nascar-predictor predict --race-date 2025-07-25

# Save predictions to file
nascar-predictor predict --output predictions.csv
```

### Weekly Updates
```bash
# Update with latest 2025 race data and retrain
nascar-predictor update-weekly --auto-retrain

# Fetch all available 2025 race data
nascar-predictor fetch-2025-data
```

## Architecture

The system uses a multi-component architecture:

1. **Data Collection**: Scrapes historical race results from Racing-Reference and speed analytics from LapRaptor.com
2. **Feature Engineering**: Creates 43+ features including performance metrics, track history, speed analytics, and momentum indicators
3. **PyTorch Models**: Tabular neural network architecture optimized for structured NASCAR data
4. **Prediction Engine**: Provides point predictions with confidence intervals and lineup optimization
5. **Incremental Learning**: Updates models with new race data automatically

## Model Features

The feature engineering pipeline creates:
- **Performance metrics**: average finish, fantasy points, consistency scores
- **Track-specific features**: track history, track type encoding
- **Speed analytics**: green flag speed, late-run performance, total speed rating
- **Momentum indicators**: recent vs historical performance trends
- **Equipment factors**: manufacturer performance, team statistics
- **Temporal features**: recent form, rolling averages

## Example Results

### Dover Motor Speedway Predictions
Recent model predictions for Dover successfully identified top performers:
1. **Denny Hamlin** (7.2 projected finish) - Won Dover in 2024
2. **Chase Elliott** (8.8 projected finish) - Excellent 2025 form
3. **Ross Chastain** (8.9 projected finish) - Strong at concrete tracks

### Sonoma Raceway Validation
Historical analysis correctly identified road course specialists:
- **Kyle Larson** (4.5 avg, winner)
- **Chase Elliott** (4.5 avg, multiple road wins)
- **Chris Buescher** (3.5 avg, road course specialist)

## Data Sources

- **Racing-Reference.info**: Historical race results, driver stats, track information
- **LapRaptor.com**: 2025 race results and detailed loop data
- **Local CSV Storage**: Normalized schema storing races, drivers, results, and analytics

## Technology Stack

- **PyTorch**: Deep learning framework for tabular neural networks
- **pandas/numpy**: Data manipulation and numerical computing
- **BeautifulSoup/requests**: Web scraping infrastructure
- **Click**: Command-line interface
- **scikit-learn**: Feature preprocessing and encoding

## Project Structure

```
nascar_fantasy_predictor/
├── cli.py                          # Enhanced CLI with subcommands
├── data/                          # Data collection and management
│   ├── csv_manager.py            # CSV-based data storage
│   ├── lapraptor_scraper.py      # LapRaptor.com data scraper
│   ├── nascar_official_scraper.py # NASCAR API integration
│   └── csv_importer.py           # Data import utilities
├── features/                      # Feature engineering
│   ├── feature_engineering.py    # Core feature creation pipeline
│   └── fantasy_points.py         # Fantasy scoring systems
├── models/                        # PyTorch model implementations
│   ├── tabular_nn.py             # Neural network architectures
│   └── trainer.py                # Training and management
└── prediction/                    # Prediction engine
    └── predictor.py              # Main prediction system
```

## Development

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Run linting
flake8

# Type checking
mypy nascar_fantasy_predictor/

# Run tests
pytest
```

## Contributing

This project demonstrates AI-assisted development workflows. While primarily AI-generated, contributions are welcome for:
- Additional data sources
- New feature engineering ideas
- Model architecture improvements
- Fantasy scoring systems

## License

MIT License - Feel free to use this as a reference for AI-assisted development projects.

## Acknowledgments

- **Claude (Anthropic)**: Primary development partner
- **NASCAR**: For providing the sport that makes this analysis possible
- **Racing-Reference & LapRaptor**: For comprehensive historical data
- **PyTorch Team**: For the deep learning framework