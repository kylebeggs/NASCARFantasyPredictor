"""Analysis and interpretation CLI commands."""

import click
from datetime import datetime, timedelta
from pathlib import Path

from ...prediction.predictor import NASCARPredictor


@click.group(name='analyze')
def analysis_group():
    """Model analysis and interpretation commands."""
    pass


@analysis_group.command()
@click.option('--model-path', help='Path to trained model')
@click.option('--output', help='Output file path for report (without extension)')
@click.option('--methods', multiple=True, default=['weights', 'gradients', 'permutation'], 
              help='Methods to use for importance analysis')
def features(model_path, output, methods):
    """Analyze and display feature importance for the trained model."""
    click.echo("Analyzing feature importance...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first or specify model path.")
            return
        
        # Get global feature importance
        click.echo("Computing feature importance (this may take a moment)...")
        global_analysis = predictor.get_global_feature_importance(top_k=20)
        
        # Display results
        click.echo(f"\nFeature Importance Analysis")
        click.echo("=" * 60)
        click.echo(f"Total features analyzed: {global_analysis['total_features']}")
        click.echo(f"Top 10 features account for: {global_analysis['top_10_contribution']:.1%} of total importance")
        
        click.echo(f"\nTop 20 Most Important Features:")
        click.echo("-" * 60)
        click.echo(f"{'Rank':<4} {'Feature':<30} {'Importance':<12} {'Category':<15}")
        click.echo("-" * 60)
        
        # Get feature categories for display
        feature_categories = global_analysis['feature_categories']
        feature_to_category = {}
        for category, features in feature_categories.items():
            for feature in features:
                feature_to_category[feature] = category
        
        for feature_info in global_analysis['top_features']:
            rank = feature_info['rank']
            feature = feature_info['feature']
            importance = feature_info['importance']
            category = feature_to_category.get(feature, 'other')
            
            click.echo(f"{rank:<4} {feature:<30} {importance:<12.4f} {category:<15}")
        
        # Show feature categories summary
        click.echo(f"\nFeature Categories:")
        click.echo("-" * 30)
        for category, features in feature_categories.items():
            click.echo(f"{category.title()}: {len(features)} features")
        
        # Save detailed report if requested
        if output:
            click.echo(f"\nGenerating comprehensive report...")
            predictor.save_feature_importance_report(output, list(methods))
            click.echo(f"Detailed report saved to {output}.*")
        
    except Exception as e:
        click.echo(f"Error analyzing feature importance: {e}")


@analysis_group.command()
@click.option('--driver', required=True, help='Driver name to explain')
@click.option('--race-date', help='Race date for prediction (YYYY-MM-DD)')
@click.option('--start-position', type=int, help='Starting position for this driver (1-40)')
@click.option('--model-path', help='Path to trained model')
@click.option('--method', default='gradients', type=click.Choice(['gradients', 'shap']),
              help='Method for explanation')
def explain(driver, race_date, start_position, model_path, method):
    """Explain prediction for a specific driver."""
    if not race_date:
        # Use next Sunday as default
        today = datetime.now()
        days_ahead = 6 - today.weekday()  # Sunday = 6
        if days_ahead <= 0:
            days_ahead += 7
        race_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    start_text = f" starting P{start_position}" if start_position else ""
    click.echo(f"Explaining prediction for {driver} on {race_date}{start_text}...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first or specify model path.")
            return
        
        # Get basic prediction first
        prediction_result = predictor.get_driver_prediction(driver, race_date, start_position)
        
        # Get explanation
        click.echo(f"Generating {method} explanation...")
        explanation = predictor.explain_prediction(driver, race_date, start_position, method=method)
        
        # Display prediction summary
        click.echo(f"\nPREDICTION EXPLANATION for {driver}")
        click.echo("=" * 50)
        click.echo(f"Race Date: {race_date}")
        click.echo(f"Starting Position: P{prediction_result['starting_position']}")
        click.echo(f"Predicted Finish: {prediction_result['predicted_finish_position']:.1f}")
        click.echo(f"Position Change: {prediction_result['predicted_position_change']:+.1f}")
        click.echo(f"Uncertainty: ±{prediction_result['prediction_uncertainty']:.1f}")
        
        # Show top contributing features
        click.echo(f"\nTop Features Contributing to Prediction ({method}):")
        click.echo("-" * 50)
        click.echo(f"{'Feature':<25} {'Contribution':<12} {'Value':<10}")
        click.echo("-" * 50)
        
        for feature_info in explanation['top_features'][:10]:
            feature = feature_info['feature'][:24]
            contribution = f"{feature_info['contribution']:+.3f}"
            value = f"{feature_info['value']:.2f}"
            click.echo(f"{feature:<25} {contribution:<12} {value:<10}")
        
        # Show interpretation
        if 'interpretation' in explanation:
            click.echo(f"\nInterpretation:")
            click.echo(f"  • {explanation['interpretation']}")
        
    except Exception as e:
        click.echo(f"Error explaining prediction: {e}")


@analysis_group.command()
@click.option('--model-path', help='Path to trained model')
@click.option('--num-drivers', default=10, help='Number of drivers to analyze')
def lineup(model_path, num_drivers):
    """Analyze optimal lineup based on value vs projected finish."""
    click.echo(f"Analyzing optimal lineup strategy for top {num_drivers} drivers...")
    
    try:
        predictor = NASCARPredictor(model_path=model_path)
        
        if predictor.trainer.model is None:
            click.echo("No trained model found. Please train a model first or specify model path.")
            return
        
        # Get lineup analysis
        analysis = predictor.analyze_lineup_strategy(num_drivers=num_drivers)
        
        if 'error' in analysis:
            click.echo(f"Analysis error: {analysis['error']}")
            return
        
        click.echo(f"\nLINEUP OPTIMIZATION ANALYSIS")
        click.echo("=" * 60)
        click.echo(f"Based on {analysis['total_drivers']} active drivers")
        
        click.echo(f"\nTop Value Picks (Best Projected Finish vs Cost):")
        click.echo("-" * 60)
        click.echo(f"{'Rank':<4} {'Driver':<20} {'Proj':<5} {'Cost':<6} {'Value':<8} {'ROI':<6}")
        click.echo("-" * 60)
        
        for i, driver in enumerate(analysis['value_picks'][:num_drivers]):
            rank = i + 1
            name = driver['driver_name'][:19]
            proj = f"{driver['projected_finish']:.1f}"
            cost = f"${driver['salary']:,}"
            value = f"{driver['value_score']:.2f}"
            roi = f"{driver['roi_estimate']:+.1%}"
            
            click.echo(f"{rank:<4} {name:<20} {proj:<5} {cost:<6} {value:<8} {roi:<6}")
        
        # Show strategy recommendations
        if 'strategy' in analysis:
            strategy = analysis['strategy']
            click.echo(f"\nStrategy Recommendations:")
            click.echo(f"  • Suggested spend: {strategy['suggested_spend']}")
            click.echo(f"  • Risk level: {strategy['risk_level']}")
            click.echo(f"  • Key insight: {strategy['key_insight']}")
        
    except Exception as e:
        click.echo(f"Error analyzing lineup: {e}")