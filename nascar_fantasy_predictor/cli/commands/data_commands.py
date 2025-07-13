"""Data management CLI commands."""

import click
import pandas as pd
from datetime import datetime
from pathlib import Path

from ...data.csv_manager import CSVDataManager
from ...data.nascar_official_scraper import NASCAROfficialScraper
from ...data.mock_data import create_mock_data_fallback
from ...data.csv_importer import CSVDataImporter
from ...data.lapraptor_scraper import LapRaptorScraper


@click.group(name='data')
def data_group():
    """Data management commands."""
    pass


@data_group.command()
@click.option('--start-year', default=2022, help='Starting year for data collection')
@click.option('--end-year', default=None, help='Ending year for data collection')
@click.option('--force', is_flag=True, help='Force re-download of existing data')
@click.option('--use-mock-data', is_flag=True, help='Use mock data instead of scraping')
@click.option('--use-real-data/--no-use-real-data', default=True, help='Use real NASCAR API data (default: True)')
def init(start_year, end_year, force, use_mock_data, use_real_data):
    """Initialize CSV data storage and download historical data."""
    if end_year is None:
        end_year = datetime.now().year
    else:
        end_year = int(end_year)
    
    click.echo(f"Initializing NASCAR Fantasy Predictor...")
    click.echo(f"Data range: {start_year} - {end_year}")
    
    # Initialize CSV data manager
    csv_manager = CSVDataManager()
    click.echo("Setting up CSV data storage...")
    
    # Download historical data
    if use_mock_data:
        click.echo("Generating mock historical race data...")
        historical_data = create_mock_data_fallback(start_year, end_year)
    elif use_real_data:
        click.echo("Fetching real NASCAR data from official API...")
        scraper = NASCAROfficialScraper()
        
        all_data = []
        for year in range(start_year, end_year + 1):
            click.echo(f"  Downloading {year} NASCAR Cup Series data...")
            try:
                year_data = scraper.get_cup_series_data(year)
                if year_data is not None and not year_data.empty:
                    all_data.append(year_data)
                    click.echo(f"    ✓ {year}: {len(year_data)} records")
                else:
                    click.echo(f"    ✗ {year}: No data available")
            except Exception as e:
                click.echo(f"    ✗ {year}: Error - {e}")
        
        if all_data:
            historical_data = pd.concat(all_data, ignore_index=True)
            click.echo(f"Total records collected: {len(historical_data)}")
        else:
            click.echo("No data collected. Falling back to mock data...")
            historical_data = create_mock_data_fallback(start_year, end_year)
    
    # Add to CSV storage
    try:
        csv_manager.add_race_results(historical_data)
        stats = csv_manager.get_data_stats()
        
        click.echo(f"\n✅ Initialization completed successfully!")
        click.echo(f"  • Total races: {stats['total_races']}")
        click.echo(f"  • Unique drivers: {stats['unique_drivers']}")
        click.echo(f"  • Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        click.echo(f"\nYou can now train a model with: nascar-predictor train model")
        
    except Exception as e:
        click.echo(f"Error during initialization: {e}")


@data_group.command()
@click.option('--nascar-file', help='Path to NASCAR formatted CSV file')
def migrate(nascar_file):
    """Migrate existing CSV files to master data format."""
    try:
        from ...prediction.predictor import NASCARPredictor
        predictor = NASCARPredictor()
        
        # Auto-detect files if not specified
        if not nascar_file:
            possible_nascar_files = [
                "nascar_formatted_data.csv",
                "nascar_data.csv", 
                "race_results.csv"
            ]
            
            for file_path in possible_nascar_files:
                if Path(file_path).exists():
                    nascar_file = file_path
                    break
        
        if not nascar_file:
            click.echo("No NASCAR CSV files found to migrate. Please specify file path.")
            return
        
        click.echo(f"Migrating data...")
        click.echo(f"  NASCAR file: {nascar_file}")
        
        records_migrated = predictor.migrate_existing_data(nascar_file)
        
        click.echo(f"Successfully migrated {records_migrated} records to master data file")
        
    except Exception as e:
        click.echo(f"Error during migration: {e}")


@data_group.command()
@click.option('--year', default=2025, help='Year to update data for')
@click.option('--auto-retrain', is_flag=True, help='Automatically retrain model after update')
@click.option('--scrape-all', is_flag=True, help='Scrape all available races for the year')
def update_weekly(year, auto_retrain, scrape_all):
    """Update database with latest race data from LapRaptor.com."""
    click.echo(f"Updating {year} race data from LapRaptor.com...")
    
    try:
        # Initialize scraper and data manager
        scraper = LapRaptorScraper()
        csv_manager = CSVDataManager()
        
        # Get current data stats
        stats = csv_manager.get_data_stats()
        if 'error' not in stats:
            click.echo(f"Current data: {stats['total_races']} races, latest: {stats['date_range']['latest']}")
        
        # Scrape 2025 data
        if scrape_all:
            click.echo("Scraping all available races...")
            new_data = scraper.scrape_all_2025_data()
        else:
            click.echo("Scraping completed races...")
            completed_races = scraper.get_completed_2025_races()
            click.echo(f"Found {len(completed_races)} completed races")
            
            all_race_data = []
            for race in completed_races:
                race_data = scraper.get_race_results(race['race_id'])
                if race_data is not None and not race_data.empty:
                    standardized_data = scraper.standardize_race_data(race_data)
                    all_race_data.append(standardized_data)
            
            if all_race_data:
                new_data = pd.concat(all_race_data, ignore_index=True)
            else:
                new_data = pd.DataFrame()
        
        if new_data.empty:
            click.echo("No new data found.")
            return
        
        # Add new data to database
        click.echo(f"Adding {len(new_data)} new records to database...")
        csv_manager.add_race_results(new_data)
        
        # Show updated stats
        updated_stats = csv_manager.get_data_stats()
        if 'error' not in updated_stats:
            click.echo(f"Updated data: {updated_stats['total_races']} races, latest: {updated_stats['date_range']['latest']}")
        
        # Auto-retrain if requested
        if auto_retrain:
            from ...prediction.predictor import NASCARPredictor
            
            click.echo("Retraining model with updated data...")
            predictor = NASCARPredictor()
            
            training_result = predictor.train_model(epochs=100, validation_split=0.2)
            
            click.echo(f"Retraining completed!")
            click.echo(f"Final validation MAE: {training_result['val_mae']:.2f}")
            click.echo(f"Training MAE: {training_result['train_mae']:.2f}")
            click.echo(f"Epochs trained: {training_result['epochs_trained']}")
            
            # Save retrained model
            output_path = f"models/nascar_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            predictor.save_model(output_path)
            click.echo(f"Model saved to {output_path}")
        
        click.echo("Weekly update completed successfully!")
        
    except Exception as e:
        click.echo(f"Error during weekly update: {e}")


@data_group.command()
@click.option('--year', default=2025, help='Year to fetch data for')
def fetch_2025(year):
    """Fetch all available 2025 race data from LapRaptor.com."""
    click.echo(f"Fetching all {year} NASCAR Cup Series data from LapRaptor.com...")
    
    try:
        # Initialize scraper and data manager
        scraper = LapRaptorScraper()
        csv_manager = CSVDataManager()
        
        # Get all 2025 data
        new_data = scraper.scrape_all_2025_data()
        
        if new_data.empty:
            click.echo("No data available for 2025.")
            return
        
        # Add to database
        click.echo(f"Adding {len(new_data)} records to database...")
        csv_manager.add_race_results(new_data)
        
        # Show stats
        stats = csv_manager.get_data_stats()
        if 'error' not in stats:
            click.echo(f"Total data: {stats['total_races']} races")
            click.echo(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            click.echo(f"Unique drivers: {stats['unique_drivers']}")
            click.echo(f"Unique tracks: {stats['unique_tracks']}")
        
        click.echo("Data fetch completed successfully!")
        
    except Exception as e:
        click.echo(f"Error fetching 2025 data: {e}")