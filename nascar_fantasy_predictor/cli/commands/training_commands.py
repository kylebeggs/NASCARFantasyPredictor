"""Training-related CLI commands."""

import click
from datetime import datetime
from pathlib import Path

from ...prediction.predictor import NASCARPredictor


@click.group(name='train')
def training_group():
    """Model training commands."""
    pass


@training_group.command()
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--learning-rate', default=0.001, help='Learning rate for training')
@click.option('--batch-size', default=32, help='Batch size for training')
@click.option('--validation-split', default=0.2, help='Validation split ratio')
@click.option('--output-path', help='Path to save trained model')
@click.option('--early-stopping-patience', default=10, help='Early stopping patience')
def model(epochs, learning_rate, batch_size, validation_split, output_path, early_stopping_patience):
    """Train a new model on historical data."""
    click.echo("Starting model training...")
    click.echo(f"Training parameters:")
    click.echo(f"  • Epochs: {epochs}")
    click.echo(f"  • Learning rate: {learning_rate}")
    click.echo(f"  • Batch size: {batch_size}")
    click.echo(f"  • Validation split: {validation_split}")
    click.echo(f"  • Early stopping patience: {early_stopping_patience}")
    
    try:
        # Initialize predictor
        predictor = NASCARPredictor()
        
        # Train model
        training_result = predictor.train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save model
        if not output_path:
            output_path = f"models/nascar_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(output_path)
        
        click.echo(f"\n✅ Training completed successfully!")
        click.echo(f"  • Training MAE: {training_result['train_mae']:.3f}")
        click.echo(f"  • Validation MAE: {training_result['val_mae']:.3f}")
        click.echo(f"  • Epochs trained: {training_result['epochs_trained']}")
        click.echo(f"  • Model saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error during training: {e}")


@training_group.command()
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


@training_group.command()
@click.option('--epochs', default=200, help='Number of training epochs')
@click.option('--learning-rate', default=0.001, help='Learning rate for training')
@click.option('--batch-size', default=16, help='Batch size for training')
@click.option('--validation-split', default=0.15, help='Validation split ratio')
@click.option('--early-stopping-patience', default=25, help='Early stopping patience')
@click.option('--output-path', help='Path to save trained model')
def aggressive(epochs, learning_rate, batch_size, validation_split, early_stopping_patience, output_path):
    """Train with aggressive parameters for maximum accuracy."""
    click.echo("🔥 Starting aggressive training for maximum accuracy...")
    click.echo(f"Aggressive parameters:")
    click.echo(f"  • Epochs: {epochs}")
    click.echo(f"  • Learning rate: {learning_rate}")
    click.echo(f"  • Batch size: {batch_size}")
    click.echo(f"  • Validation split: {validation_split}")
    click.echo(f"  • Early stopping patience: {early_stopping_patience}")
    
    try:
        predictor = NASCARPredictor()
        
        training_result = predictor.train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience
        )
        
        if not output_path:
            output_path = f"models/aggressive_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(output_path)
        
        val_mae = training_result['val_mae']
        if val_mae < 7.0:
            status = "🎉 EXCELLENT"
        elif val_mae < 8.5:
            status = "✅ GOOD"
        else:
            status = "⚠️ ACCEPTABLE"
        
        click.echo(f"\n{status}: Aggressive training completed!")
        click.echo(f"  • Training MAE: {training_result['train_mae']:.3f}")
        click.echo(f"  • Validation MAE: {training_result['val_mae']:.3f}")
        click.echo(f"  • Epochs trained: {training_result['epochs_trained']}")
        click.echo(f"  • Model saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error during aggressive training: {e}")