# 🏁 NASCAR Fantasy Predictor 🤖

![CI](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/CI/badge.svg)
![Tests](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/Tests/badge.svg)
![Coverage](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/Test%20Coverage/badge.svg)

> **⚠️ DISCLAIMER & WARNING ⚠️**  
> This repository is 99% written by Claude (Anthropic's AI assistant) - I just provided the NASCAR knowledge and said "make it go fast!" 🏎️💨 This is a collaborative experiment in AI-assisted development, which means:
> 
> - 🤖 **Claude wrote the code** - Any bugs, questionable design decisions, or "why did they do it that way??" moments should be blamed entirely on the AI, not me!
> - 🧠 **I provided the NASCAR expertise** - If the predictions are wrong, that's probably my fault for not explaining why Kyle Larson is magic on dirt tracks
> - 🚀 **This is an experiment** - We're exploring what happens when you combine human domain knowledge with AI coding superpowers
> - 🎯 **Don't judge my coding skills** - I can barely write a for loop without Stack Overflow, but Claude here can apparently build entire neural networks before breakfast!
>
> Consider this a "hold my beer and watch this" moment in AI development! 🍺⚡

## 🎬 Overview

NASCAR Fantasy Predictor is a data collection and analysis tool designed to help you make informed fantasy NASCAR decisions. It focuses on gathering historical race data from multiple sources and providing insights through statistical analysis and feature engineering. This is a human-augmented decision-making tool that combines data-driven insights with your NASCAR expertise.

## ✨ Current Features

- 🕷️ **Data Collection**: Web scraping from LapRaptor.com for 2025 race results and speed analytics
- 🏁 **Race Analysis**: Detailed analysis of race results with fantasy points calculations
- 📊 **Driver Statistics**: Comprehensive driver performance metrics and historical data
- 💰 **Fantasy Points**: DraftKings scoring system with top performer insights
- 📈 **Feature Engineering**: Advanced statistical features for analysis
- 📁 **Data Export**: CSV and JSON export capabilities for external analysis
- 🎯 **Track-Specific Analysis**: Historical performance at specific tracks

## 🚀 Quick Start

### 📦 Installation
```bash
# Install the package in development mode with dev dependencies
pip install -e ".[dev]"
```

### 🏁 Data Collection
```bash
# Collect 2025 NASCAR race data from LapRaptor.com
nascar-predictor collect

# Currently only 2025 data collection is supported
nascar-predictor collect --year 2025
```

### 📊 Data Analysis
```bash
# Analyze the latest race
nascar-predictor analyze --date latest

# Analyze a specific race date
nascar-predictor analyze --date 2025-02-25

# Analyze historical data for a specific track
nascar-predictor analyze --track "Daytona"
```

### 🏎️ Driver Statistics
```bash
# Get comprehensive driver statistics
nascar-predictor driver-stats --driver "Kyle Larson"

# Driver stats for a specific track
nascar-predictor driver-stats --driver "Kyle Larson" --track "Phoenix"

# Driver stats for a specific year
nascar-predictor driver-stats --driver "Kyle Larson" --year 2024
```

### 📁 Data Export
```bash
# Export all data to CSV
nascar-predictor export --output all_races.csv

# Export specific year data
nascar-predictor export --output 2024_races.csv --year 2024

# Export as JSON format
nascar-predictor export --output races.json --format json
```

## 🏗️ Architecture Overview

The system is designed for simplicity and data analysis:

1. 🕸️ **Data Collection**: Scrapes historical race results and speed analytics from LapRaptor.com
2. 🔧 **Feature Engineering**: Creates analytical features for human interpretation and decision-making
3. 📊 **Data Storage**: Simple CSV-based storage for easy access and manipulation
4. 📈 **Analysis Tools**: Provides insights on driver performance, track history, and head-to-head matchups
5. 📁 **Export Functionality**: Easy export to CSV/JSON for external analysis tools

## 🎛️ Analysis Features

The tool provides comprehensive data analysis including:
- 📊 **Driver Performance**: Average finish, consistency scores, track-specific performance
- 🏁 **Track History**: Historical performance patterns at specific tracks
- 💨 **Speed Analytics**: Green flag speed, late-run performance, total speed ratings from LapRaptor
- 📈 **Position Analysis**: Position changes, biggest movers, starting vs finishing positions
- 💰 **Fantasy Points**: DraftKings scoring calculations and top performer identification
- ⏰ **Recent Form**: Recent race results and performance trends

## 📊 Data Sources

- ⚡ **LapRaptor.com**: 2025 race results and speed analytics including green flag speed and late-run performance
- 🏁 **NASCAR Official Site**: Official race results and statistics (scrapers available but not currently active)
- 💾 **Local CSV Storage**: Simple file-based storage for collected data

## 🛠️ Tech Stack

- 🐼 **pandas/numpy**: Data manipulation and analysis
- 🕷️ **BeautifulSoup/requests**: Web scraping for data collection
- 🖱️ **Click**: Command-line interface framework
- 📊 **tqdm**: Progress bars for data collection operations

## 🗂️ Project Structure

```
nascar_fantasy_predictor/          # Main package directory
├── cli/                          # Command line interface
│   └── main.py                  # CLI entry point with focused commands
├── data/                         # Data collection and management
│   ├── csv_manager.py           # CSV-based data storage and retrieval
│   ├── lapraptor_scraper.py     # LapRaptor.com speed analytics scraper
│   ├── nascar_official_scraper.py # NASCAR official site scraper
│   ├── qualifying_scraper.py    # Qualifying results scraper
│   └── csv_importer.py          # Data import utilities
├── features/                     # Feature engineering for analysis
│   ├── feature_engineering.py   # Data analysis feature creation
│   └── fantasy_points.py        # Fantasy points calculation (DraftKings, FanDuel)
└── utils/                        # Utility modules
    ├── logging.py               # Centralized logging configuration
    └── exceptions.py            # Custom exception classes
```

## 🛠️ Development

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

## 🤝 Contributing

This project welcomes contributions for:
- 📡 Additional data sources and scraping capabilities
- 🔧 New analysis features and statistical insights
- 📊 Enhanced visualization and reporting tools
- 💰 Support for additional fantasy scoring systems

## 📜 License

MIT License

## 🙏 Acknowledgments

- 🤖 **Claude (Anthropic)**: AI assistant for code development and architecture design
- 🏁 **NASCAR**: For creating the most exciting motorsport
- 📊 **LapRaptor.com**: For providing comprehensive speed analytics and race data
- 🏆 **NASCAR Fantasy Community**: For inspiring data-driven fantasy decision making

## 🎯 Usage Philosophy

This tool is designed to augment human decision-making, not replace it. It provides data and insights that help identify:

- Drivers with strong recent form
- Track-specific specialists  
- Consistent performers vs high-variance drivers
- Value plays based on starting position
- Historical performance patterns

The human user combines these insights with additional factors like weather conditions, team changes, equipment updates, driver motivations, and current point standings. This combination of data-driven insights and human judgment leads to better fantasy predictions than either approach alone.