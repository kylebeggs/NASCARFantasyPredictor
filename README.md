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

Welcome to the ultimate AI-powered NASCAR Fantasy Predictor! 🚀 This bad boy uses cutting-edge machine learning to help you dominate your fantasy league by predicting race finishes with scary accuracy. We're talking deep learning models trained on years of racing data, real-time analytics, and enough NASCAR knowledge to make even Dale Jr. proud! 🏆

## ✨ Features That'll Make You Go "Boogity Boogity Boogity!"

- 🕷️ **Web Scraping Wizardry**: Automatically grabs race data from Racing-Reference and LapRaptor.com like a pit crew on Red Bull
- 🧠 **Big Brain Feature Engineering**: Creates 43+ features that would make a NASA engineer jealous (speed, track history, momentum, you name it!)
- 🔥 **PyTorch Power**: Deep learning models that are faster than Kyle Larson on a superspeedway
- 🔄 **Auto-Updates**: Weekly data refreshes with model retraining - fresher than new tires on pit road
- 🏁 **Track-Specific Smarts**: Knows the difference between Daytona and Dover better than your favorite NASCAR commentator
- 💰 **Fantasy Gold**: Predictions with confidence intervals to help you win that office pool and earn bragging rights

## 🚀 Quick Start (Let's Go Racing!)

### 📦 Installation
```bash
# Get this baby installed faster than a NASCAR pit stop!
pip install -e ".[dev]"
```

### 🏁 Initial Setup
```bash
# Download years of racing history (the good stuff!)
nascar-predictor init --start-year 2022

# See what we're working with
nascar-predictor status
```

### 🧠 Train Your AI Crew Chief
```bash
# Teach the AI everything about NASCAR racing
nascar-predictor train
```

### 🎯 Make Some Predictions
```bash
# Who's gonna win this Sunday? Let's find out!
nascar-predictor predict

# Planning ahead for a specific race
nascar-predictor predict --race-date 2025-07-25

# Save those golden predictions
nascar-predictor predict --output my-winning-picks.csv
```

### 🔄 Stay Fresh with Weekly Updates
```bash
# Keep your model sharper than a fresh set of Goodyears
nascar-predictor update-weekly --auto-retrain

# Grab all the 2025 race data
nascar-predictor fetch-2025-data
```

## 🏗️ Under the Hood (The Technical Stuff)

This beast runs on a multi-component architecture that's more complex than a championship car setup:

1. 🕸️ **Data Collection Squad**: Web scrapers that work harder than a pit crew, grabbing data from Racing-Reference and LapRaptor.com
2. 🔧 **Feature Engineering Garage**: Creates 43+ mind-blowing features from speed metrics to track history 
3. 🧠 **PyTorch Powerhouse**: Neural networks that think faster than you can say "checkered flag"
4. 🎯 **Prediction Engine**: Spits out finish predictions with confidence levels that'll make you a fantasy legend
5. 📈 **Auto-Learning**: Gets smarter every week like a veteran driver learning a new track

## 🎛️ What Makes This Thing Tick

Our AI crew chief analyzes more data than a NASCAR telemetry system:
- 📊 **Performance Wizardry**: Average finishes, consistency scores, fantasy points galore
- 🏁 **Track Intelligence**: Knows every bump, bank, and characteristic of each track
- 💨 **Speed Secrets**: Green flag speed, late-run performance, total speed ratings
- 📈 **Momentum Magic**: Recent form vs historical trends (is this driver heating up or cooling down?)
- 🏎️ **Equipment Edge**: Manufacturer performance, team stats, all the garage secrets
- ⏰ **Time Travel**: Rolling averages and recent form that predict the future

## 📊 Where We Get the Good Stuff

- 🏁 **Racing-Reference.info**: The holy grail of historical NASCAR data
- ⚡ **LapRaptor.com**: Fresh 2025 race results and loop data that's hotter than asphalt in July
- 💾 **Local CSV Storage**: Our own private database that's organized better than Hendrick Motorsports' garage

## 🛠️ Tech Stack (The Nerd Stuff)

- 🔥 **PyTorch**: AI framework that's more powerful than a restrictor plate-free engine
- 🐼 **pandas/numpy**: Data crunching tools that work faster than a pit stop
- 🕷️ **BeautifulSoup/requests**: Web scraping magic that grabs data like it's on pole position
- 🖱️ **Click**: Command-line interface smoother than a freshly paved track
- 🤖 **scikit-learn**: Feature preprocessing that's more precise than laser tech inspection

## 🗂️ Project Structure (The Garage Layout)

```
nascar_fantasy_predictor/          # 🏠 Home sweet home
├── cli.py                         # 🎤 Command center for all the magic
├── data/                         # 📁 Data collection headquarters
│   ├── csv_manager.py           # 📊 Data storage that never crashes
│   ├── lapraptor_scraper.py     # 🕷️ Web scraping superhero
│   ├── nascar_official_scraper.py # 🏁 Official NASCAR data pipeline
│   └── csv_importer.py          # 📥 Import wizard
├── features/                     # ⚙️ Feature engineering workshop
│   ├── feature_engineering.py   # 🔧 The main feature factory
│   └── fantasy_points.py        # 💰 Fantasy scoring genius
├── models/                       # 🧠 AI brain center
│   ├── tabular_nn.py            # 🤖 Neural network architectures
│   └── trainer.py               # 🏋️ Model training gym
└── prediction/                   # 🔮 Crystal ball department
    └── predictor.py             # 🎯 The main prediction engine
```

## 🛠️ Development (For the Code Warriors)

### 🧹 Keep It Clean
```bash
# Make your code prettier than a freshly waxed race car
black .
isort .
flake8
mypy nascar_fantasy_predictor/
pytest
```

## 🤝 Contributing (Join the Team!)

This project is a showcase of AI-assisted development magic! 🪄 While Claude did most of the heavy lifting, we'd love contributions for:
- 📡 More data sources (the more the merrier!)
- 🔧 Crazy new feature ideas (think outside the pit box!)
- 🧠 Model improvements (make it even smarter!)
- 💰 Fantasy scoring systems (help everyone win!)

## 📜 License

MIT License - Go wild with it! Use this as inspiration for your own AI-assisted projects! 🚀

## 🙏 Shoutouts

- 🤖 **Claude (Anthropic)**: The AI co-pilot who made this whole thing possible
- 🏁 **NASCAR**: For creating the most exciting sport on the planet
- 📊 **Racing-Reference & LapRaptor**: For being the data heroes we needed
- 🔥 **PyTorch Team**: For building the AI framework that powers our predictions
- 🏆 **You**: For checking out this wild ride of AI + NASCAR!