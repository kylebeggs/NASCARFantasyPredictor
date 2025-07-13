"""Prediction-related CLI commands."""

import click
import pandas as pd
from datetime import datetime
from pathlib import Path

from ...prediction.predictor import NASCARPredictor


@click.group(name='predict')
def prediction_group():
    """Race prediction commands."""
    pass


@prediction_group.command()
@click.option('--race-date', help='Date of race (YYYY-MM-DD). If not provided, tries to predict today\'s race')
@click.option('--num-drivers', default=15, help='Number of drivers to predict')
@click.option('--model-path', help='Path to trained model')
@click.option('--output', help='Output file for predictions (CSV format)')
@click.option('--qualifying-file', help='CSV file with qualifying results (columns: driver_name, start_position)')
def race(race_date, num_drivers, model_path, output, qualifying_file):
    """Make race predictions using qualifying results."""
    
    if not race_date:
        race_date = datetime.now().strftime('%Y-%m-%d')
        click.echo(f"No race date provided, using today: {race_date}")
    
    click.echo(f"Making predictions for race on {race_date}...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first using 'nascar-predictor train'")
            return
        
        # Get qualifying results
        if qualifying_file:
            click.echo(f"Loading qualifying results from {qualifying_file}")
            qualifying_results = pd.read_csv(qualifying_file)
        else:
            click.echo("Error: Qualifying results file required. Use --qualifying-file option")
            return
        
        # Make predictions
        predictions = predictor.predict_next_race(race_date, qualifying_results, num_drivers)
        
        # Display results
        click.echo(f"\nRACE PREDICTIONS for {race_date}")
        click.echo("=" * 60)
        click.echo(f"{'Rank':<4} {'Driver':<20} {'Start':<5} {'Pred':<5} {'Change':<6} {'Uncertainty':<11}")
        click.echo("-" * 60)
        
        for i, row in predictions.iterrows():
            rank = i + 1
            driver = row['driver_name'][:19]  # Truncate long names
            start_pos = int(row.get('qualifying_position', row.get('start_position', 0)))
            pred_pos = f"{row['predicted_finish_position']:.1f}"
            change = f"{row.get('predicted_position_change', 0):+.1f}"
            uncertainty = f"±{row['prediction_uncertainty']:.1f}"
            
            click.echo(f"{rank:<4} {driver:<20} P{start_pos:<4} {pred_pos:<5} {change:<6} {uncertainty:<11}")
        
        # Save to file if requested
        if output:
            predictions.to_csv(output, index=False)
            click.echo(f"\nPredictions saved to {output}")
        
    except Exception as e:
        click.echo(f"Error making predictions: {e}")


@prediction_group.command()
@click.option('--driver-name', required=True, help='Driver name to predict')
@click.option('--race-date', help='Date of race (YYYY-MM-DD)')
@click.option('--start-position', type=int, required=True, help='Starting position for the driver')
@click.option('--model-path', help='Path to trained model')
def driver(driver_name, race_date, start_position, model_path):
    """Get detailed prediction for a specific driver."""
    
    if not race_date:
        race_date = datetime.now().strftime('%Y-%m-%d')
        click.echo(f"No race date provided, using today: {race_date}")
    
    click.echo(f"Getting prediction for {driver_name} on {race_date}...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first using 'nascar-predictor train'")
            return
        
        # Get prediction
        prediction = predictor.get_driver_prediction(driver_name, race_date, start_position)
        
        # Display results
        click.echo(f"\nPREDICTION for {driver_name}")
        click.echo("=" * 40)
        click.echo(f"Race Date: {race_date}")
        click.echo(f"Starting Position: P{prediction['starting_position']}")
        click.echo(f"Predicted Finish: {prediction['predicted_finish_position']:.1f}")
        click.echo(f"Position Change: {prediction['predicted_position_change']:+.1f}")
        click.echo(f"Confidence: ±{prediction['prediction_uncertainty']:.1f}")
        click.echo(f"Context: {prediction.get('start_position_context', 'N/A')}")
        
        if 'feature_summary' in prediction:
            click.echo(f"\nKey Factors:")
            summary = prediction['feature_summary']
            if summary.get('avg_finish'):
                click.echo(f"  • Historical Avg Finish: {summary['avg_finish']:.1f}")
            if summary.get('form_score'):
                click.echo(f"  • Recent Form Score: {summary['form_score']:+.1f}")
        
    except Exception as e:
        click.echo(f"Error getting driver prediction: {e}")


@prediction_group.command()
@click.option('--race-date', required=True, help='Date of race to evaluate (YYYY-MM-DD)')
@click.option('--model-path', help='Path to trained model')
@click.option('--qualifying-file', help='CSV file with qualifying results')
def evaluate(race_date, model_path, qualifying_file):
    """Evaluate prediction accuracy against actual race results."""
    click.echo(f"Evaluating predictions for race on {race_date}...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please specify model path or train a model first.")
            return
        
        # Get qualifying results
        if qualifying_file:
            qualifying_results = pd.read_csv(qualifying_file)
        else:
            click.echo("Error: Qualifying results required for evaluation. Use --qualifying-file option.")
            return
        
        evaluation = predictor.evaluate_predictions(race_date, qualifying_results)
        
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