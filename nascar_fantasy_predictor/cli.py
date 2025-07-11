"""Command line interface for NASCAR Fantasy Predictor."""

import click
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from .data.csv_manager import CSVDataManager
from .data.nascar_official_scraper import NASCAROfficialScraper
from .data.mock_data import create_mock_data_fallback
from .data.csv_importer import CSVDataImporter
from .data.lapraptor_scraper import LapRaptorScraper
from .prediction.predictor import NASCARPredictor


@click.group()
@click.version_option()
def cli():
    """NASCAR Fantasy Predictor - AI-powered fantasy league picks."""
    pass


@cli.command()
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
        
        try:
            historical_data = scraper.get_historical_data(start_year, end_year, series_id=1)
            
            if historical_data.empty:
                click.echo("No real data available. Falling back to mock data...")
                historical_data = create_mock_data_fallback(start_year, end_year)
            else:
                click.echo(f"Successfully fetched {len(historical_data)} real NASCAR records!")
        except Exception as e:
            click.echo(f"Error fetching real data: {e}")
            click.echo("Falling back to mock data...")
            historical_data = create_mock_data_fallback(start_year, end_year)
    else:
        click.echo("Only real NASCAR data and mock data are supported.")
        click.echo("Falling back to mock data...")
        historical_data = create_mock_data_fallback(start_year, end_year)
    
    if historical_data.empty:
        click.echo("Failed to generate any data.")
        return
    
    try:
        # Store race data in CSV files
        csv_manager.add_race_results(historical_data)
        
        click.echo(f"Successfully processed {len(historical_data)} race results")
        click.echo("Initialization complete!")
        
    except Exception as e:
        click.echo(f"Error during initialization: {e}")


@cli.command()
@click.option('--csv-file', required=True, help='Path to CSV file with race data')
@click.option('--data-type', default='race_results', type=click.Choice(['race_results', 'lap_data']), help='Type of data in CSV')
def import_csv(csv_file, data_type):
    """Import race data from CSV file."""
    try:
        csv_manager = CSVDataManager()
        
        if data_type == 'race_results':
            click.echo(f"Importing race results from {csv_file}...")
            data = pd.read_csv(csv_file)
            csv_manager.add_race_results(data)
            click.echo(f"Successfully imported {len(data)} race results")
            
        elif data_type == 'lap_data':
            click.echo(f"Importing lap-by-lap data from {csv_file}...")
            data = pd.read_csv(csv_file)
            click.echo(f"Successfully imported {len(data)} lap records")
            click.echo("Note: Lap data storage not yet implemented - coming soon!")
        
    except Exception as e:
        click.echo(f"Error importing CSV: {e}")


@cli.command()
@click.option('--output-dir', default='.', help='Directory to create CSV templates')
def create_templates(output_dir):
    """Create sample CSV templates for data import."""
    try:
        importer = CSVDataImporter()
        importer.create_sample_csv_templates(output_dir)
        click.echo("CSV templates created successfully!")
    except Exception as e:
        click.echo(f"Error creating templates: {e}")


@cli.command()
@click.option('--race-date', help='Date of race to predict (YYYY-MM-DD)')
@click.option('--num-drivers', default=20, help='Number of top drivers to show')
@click.option('--output', help='Output file path (JSON or CSV)')
@click.option('--model-path', help='Path to trained model')
def predict(race_date, num_drivers, output, model_path):
    """Generate fantasy point predictions for upcoming race."""
    if not race_date:
        # Use next Sunday as default
        today = datetime.now()
        days_ahead = 6 - today.weekday()  # Sunday = 6
        if days_ahead <= 0:
            days_ahead += 7
        race_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    click.echo(f"Generating predictions for race on {race_date}...")
    
    try:
        # Auto-detect latest model if no path specified
        if not model_path:
            model_files = list(Path("models").glob("nascar_model_*.pth"))
            if model_files:
                # Get the most recent model file
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                model_path = str(latest_model).replace('.pth', '')
                click.echo(f"Using latest model: {model_path}")
        
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first with 'nascar-predictor train'")
            return
        
        predictions = predictor.predict_next_race(race_date, num_drivers)
        
        # Display results
        click.echo(f"\nTop {len(predictions)} Finish Position Predictions for {race_date}:")
        click.echo("-" * 80)
        click.echo(f"{'Rank':<4} {'Driver':<25} {'Predicted Position':<15} {'Confidence':<15} {'Team':<15}")
        click.echo("-" * 80)
        
        for i, row in predictions.iterrows():
            confidence = f"±{row['prediction_uncertainty']:.1f}"
            team = row.get('team') or 'N/A'
            click.echo(f"{i+1:<4} {row['driver_name']:<25} {row['predicted_finish_position']:<15.1f} "
                      f"{confidence:<15} {team:<15}")
        
        # Save output if requested
        if output:
            output_path = Path(output)
            if output_path.suffix.lower() == '.json':
                predictions.to_json(output_path, orient='records', indent=2)
            else:
                predictions.to_csv(output_path, index=False)
            click.echo(f"\nPredictions saved to {output_path}")
        
    except Exception as e:
        click.echo(f"Error generating predictions: {e}")


@cli.command()
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--output-path', help='Path to save trained model')
@click.option('--validation-split', default=0.2, help='Validation data split ratio')
def train(epochs, output_path, validation_split):
    """Train the prediction model on all available historical data."""
    click.echo(f"Training tabular model with all available data...")
    
    try:
        predictor = NASCARPredictor()
        
        training_result = predictor.train_model(
            epochs=epochs,
            validation_split=validation_split
        )
        
        click.echo(f"Training completed!")
        click.echo(f"Final validation MAE: {training_result['val_mae']:.2f}")
        click.echo(f"Training MAE: {training_result['train_mae']:.2f}")
        click.echo(f"Epochs trained: {training_result['epochs_trained']}")
        
        # Save model
        if not output_path:
            output_path = f"models/nascar_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(output_path)
        
    except Exception as e:
        click.echo(f"Error during training: {e}")


@cli.command()
@click.option('--since-date', required=True, help='Update with data since this date (YYYY-MM-DD)')
@click.option('--model-path', required=True, help='Path to existing model to update')
def update(since_date, model_path):
    """Update trained model with new race data."""
    click.echo(f"Updating model with data since {since_date}...")
    
    try:
        # Update model
        predictor = NASCARPredictor(model_path=model_path)
        predictor.update_model(since_date)
        
        # Save updated model
        predictor.save_model(model_path)
        click.echo("Model update completed!")
        
    except Exception as e:
        click.echo(f"Error during update: {e}")



@cli.command()
@click.option('--race-date', required=True, help='Date of race to evaluate (YYYY-MM-DD)')
@click.option('--model-path', help='Path to trained model')
def evaluate(race_date, model_path):
    """Evaluate prediction accuracy against actual race results."""
    click.echo(f"Evaluating predictions for race on {race_date}...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please specify model path or train a model first.")
            return
        
        evaluation = predictor.evaluate_predictions(race_date)
        
        if 'error' in evaluation:
            click.echo(f"Evaluation error: {evaluation['error']}")
            return
        
        click.echo(f"\nPrediction Accuracy for {race_date}:")
        click.echo("-" * 40)
        click.echo(f"Mean Absolute Error: {evaluation['mae']:.2f} points")
        click.echo(f"Root Mean Square Error: {evaluation['rmse']:.2f} points")
        click.echo(f"Correlation: {evaluation['correlation']:.3f}")
        click.echo(f"Top 10 Accuracy: {evaluation['top10_accuracy']*100:.1f}%")
        click.echo(f"Drivers Evaluated: {evaluation['drivers_evaluated']}")
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")


@cli.command()
@click.option('--model-path', help='Path to trained model')
def status(model_path):
    """Show model status and performance metrics."""
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        # Show model info
        click.echo("NASCAR Fantasy Predictor Status")
        click.echo("-" * 40)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found.")
        else:
            click.echo(f"Model Type: tabular")
            click.echo(f"Device: {predictor.trainer.device}")
            
            if predictor.trainer.training_history:
                latest = predictor.trainer.training_history[-1]
                click.echo(f"Last Training Validation MAE: {latest['val_mae']:.2f}")
                click.echo(f"Training Epochs: {latest['epochs_trained']}")
        
        # Data statistics
        stats = predictor.get_data_stats()
        
        if 'error' not in stats:
            click.echo(f"\nData Statistics:")
            click.echo(f"Total Race Results: {stats['total_races']}")
            click.echo(f"Unique Drivers: {stats['unique_drivers']}")
            click.echo(f"Unique Tracks: {stats['unique_tracks']}")
            click.echo(f"Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            click.echo(f"Series: {stats['series']}")
        else:
            click.echo(f"\nData Error: {stats['error']}")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}")


@cli.command()
@click.option('--nascar-file', help='Path to NASCAR formatted CSV file')
def migrate(nascar_file):
    """Migrate existing CSV files to master data format."""
    try:
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


@cli.command()
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


@cli.command()
@click.option('--year', default=2025, help='Year to fetch data for')
def fetch_2025_data(year):
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


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()