"""Command line interface for NASCAR Fantasy Predictor."""

import click
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from .data.database import DatabaseManager
from .data.nascar_official_scraper import NASCAROfficialScraper
from .data.mock_data import create_mock_data_fallback
from .data.csv_importer import CSVDataImporter
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
    """Initialize database and download historical data."""
    if end_year is None:
        end_year = datetime.now().year
    else:
        end_year = int(end_year)
    
    click.echo(f"Initializing NASCAR Fantasy Predictor...")
    click.echo(f"Data range: {start_year} - {end_year}")
    
    # Initialize database
    db_manager = DatabaseManager()
    click.echo("Setting up database...")
    db_manager.initialize_database()
    
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
        # Store race data in database
        with db_manager.get_connection() as conn:
            db_manager.insert_race_results_from_dataframe(historical_data, 'race_results', conn)
        
        click.echo(f"Successfully processed {len(historical_data)} race results")
        click.echo("Initialization complete!")
        
    except Exception as e:
        click.echo(f"Error during initialization: {e}")


@cli.command()
@click.option('--csv-file', required=True, help='Path to CSV file with race data')
@click.option('--data-type', default='race_results', type=click.Choice(['race_results', 'lap_data', 'lapraptor']), help='Type of data in CSV')
def import_csv(csv_file, data_type):
    """Import race data from CSV file."""
    try:
        importer = CSVDataImporter()
        db_manager = DatabaseManager()
        
        if data_type == 'race_results':
            click.echo(f"Importing race results from {csv_file}...")
            data = importer.import_race_results(csv_file)
            
            with db_manager.get_connection() as conn:
                db_manager.insert_race_results_from_dataframe(data, 'race_results', conn)
            
            click.echo(f"Successfully imported {len(data)} race results")
            
        elif data_type == 'lap_data':
            click.echo(f"Importing lap-by-lap data from {csv_file}...")
            data = importer.import_lap_by_lap_data(csv_file)
            click.echo(f"Successfully imported {len(data)} lap records")
            click.echo("Note: Lap data storage not yet implemented - coming soon!")
        elif data_type == 'lapraptor':
            click.echo(f"Importing lapraptor data from {csv_file}...")
            data = pd.read_csv(csv_file)
            with db_manager.get_connection() as conn:
                db_manager.insert_race_results_from_dataframe(data, 'lapraptor', conn)
            click.echo(f"Successfully imported {len(data)} lapraptor records")
        
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
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first with 'nascar-predictor train'")
            return
        
        predictions = predictor.predict_next_race(race_date, num_drivers)
        
        # Display results
        click.echo(f"\nTop {len(predictions)} Fantasy Point Predictions for {race_date}:")
        click.echo("-" * 80)
        click.echo(f"{'Rank':<4} {'Driver':<25} {'Predicted Points':<15} {'Confidence':<15} {'Team':<15}")
        click.echo("-" * 80)
        
        for i, row in predictions.iterrows():
            confidence = f"±{row['prediction_uncertainty']:.1f}"
            click.echo(f"{i+1:<4} {row['driver_name']:<25} {row['predicted_fantasy_points']:<15.1f} "
                      f"{confidence:<15} {row.get('team', 'N/A'):<15}")
        
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
@click.option('--start-date', required=True, help='Start date for training data (YYYY-MM-DD)')
@click.option('--end-date', help='End date for training data (YYYY-MM-DD)')
@click.option('--model-type', default='nascar', type=click.Choice(['tabular', 'ensemble', 'nascar']))
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--output-path', help='Path to save trained model')
@click.option('--validation-split', default=0.2, help='Validation data split ratio')
def train(start_date, end_date, model_type, epochs, output_path, validation_split):
    """Train the prediction model on historical data."""
    click.echo(f"Training {model_type} model with data from {start_date} to {end_date or 'present'}...")
    
    try:
        predictor = NASCARPredictor()
        
        training_result = predictor.train_model(
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
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
        # First, collect new data
        scraper = RacingReferenceScraper()
        new_data = scraper.get_historical_data(
            datetime.strptime(since_date, '%Y-%m-%d').year,
            datetime.now().year
        )
        
        if not new_data.empty:
            # Process and store new data (similar to init)
            db_manager = DatabaseManager()
            calculator = FantasyPointsCalculator()
            conn = db_manager.get_connection()
            
            # Store new data (implementation similar to init command)
            # ... (data storage code)
            
            conn.close()
        
        # Update model
        predictor = NASCARPredictor(model_path=model_path)
        predictor.update_model(since_date)
        
        # Save updated model
        predictor.save_model(model_path)
        click.echo("Model update completed!")
        
    except Exception as e:
        click.echo(f"Error during update: {e}")


@cli.command()
@click.option('--year', default=datetime.now().year, help='Year to fetch loop data for')
def update_lap_data(year):
    """Fetch loop data from lapraptor.com and save to CSV."""
    try:
        from .data.lapraptor_scraper import LapRaptorScraper
        
        scraper = LapRaptorScraper()
        df = scraper.get_loop_data(year)
        
        if not df.empty:
            output_path = f"lapraptor_{year}.csv"
            df.to_csv(output_path, index=False)
            click.echo(f"Successfully saved loop data to {output_path}")
        else:
            click.echo(f"No loop data found for {year}")
            
    except Exception as e:
        click.echo(f"Error updating lap data: {e}")


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
        
        if predictor.trainer.model is None:
            click.echo("No trained model found.")
            return
        
        # Show model info
        click.echo("NASCAR Fantasy Predictor Status")
        click.echo("-" * 40)
        click.echo(f"Model Type: {predictor.trainer.model_type}")
        click.echo(f"Device: {predictor.trainer.device}")
        
        if predictor.trainer.training_history:
            latest = predictor.trainer.training_history[-1]
            click.echo(f"Last Training Validation MAE: {latest['val_mae']:.2f}")
            click.echo(f"Training Epochs: {latest['epochs_trained']}")
        
        # Database stats
        conn = predictor.db_connection
        cursor = conn.execute("SELECT COUNT(*) FROM races")
        race_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM race_results")
        result_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT driver_id) FROM race_results")
        driver_count = cursor.fetchone()[0]
        
        click.echo(f"\nDatabase Statistics:")
        click.echo(f"Races: {race_count}")
        click.echo(f"Race Results: {result_count}")
        click.echo(f"Unique Drivers: {driver_count}")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()