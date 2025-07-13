"""Maintenance and utility CLI commands."""

import click
import pandas as pd
from datetime import datetime
from pathlib import Path

from ...data.csv_manager import CSVDataManager
from ...prediction.predictor import NASCARPredictor


@click.group(name='system')
def maintenance_group():
    """System maintenance and utility commands."""
    pass


@maintenance_group.command()
def status():
    """Check system status and data availability."""
    click.echo("NASCAR Fantasy Predictor System Status")
    click.echo("=" * 40)
    
    try:
        # Check data status
        csv_manager = CSVDataManager()
        stats = csv_manager.get_data_stats()
        
        if 'error' in stats:
            click.echo(f"❌ Data Status: {stats['error']}")
            click.echo("   Run 'nascar-predictor data init' to initialize data")
        else:
            click.echo(f"✅ Data Status: Active")
            click.echo(f"   • Total races: {stats['total_races']}")
            click.echo(f"   • Unique drivers: {stats['unique_drivers']}")
            click.echo(f"   • Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            click.echo(f"   • Unique tracks: {stats['unique_tracks']}")
        
        # Check model status
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pth"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                click.echo(f"✅ Model Status: Available")
                click.echo(f"   • Latest model: {latest_model.name}")
                click.echo(f"   • Modified: {datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
            else:
                click.echo(f"⚠️ Model Status: No trained models found")
                click.echo("   Run 'nascar-predictor train model' to train a model")
        else:
            click.echo(f"⚠️ Model Status: Models directory not found")
            click.echo("   Run 'nascar-predictor train model' to train a model")
        
        click.echo(f"\n📊 System Ready: {'Yes' if 'error' not in stats and model_files else 'No'}")
        
    except Exception as e:
        click.echo(f"Error checking status: {e}")


@maintenance_group.command()
@click.option('--backup-location', default='backups/', help='Directory to store backups')
def backup():
    """Create backup of data and models."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(backup_location) / f"backup_{timestamp}"
    
    click.echo(f"Creating backup in {backup_dir}...")
    
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup data files
        data_files = ['master_race_data.csv', 'nascar_formatted_data.csv']
        data_backed_up = 0
        
        for data_file in data_files:
            if Path(data_file).exists():
                backup_path = backup_dir / data_file
                backup_path.write_bytes(Path(data_file).read_bytes())
                data_backed_up += 1
        
        # Backup models
        models_dir = Path("models")
        models_backed_up = 0
        
        if models_dir.exists():
            model_backup_dir = backup_dir / "models"
            model_backup_dir.mkdir(exist_ok=True)
            
            for model_file in models_dir.glob("*.pth"):
                backup_path = model_backup_dir / model_file.name
                backup_path.write_bytes(model_file.read_bytes())
                models_backed_up += 1
        
        click.echo(f"✅ Backup completed successfully!")
        click.echo(f"   • Data files backed up: {data_backed_up}")
        click.echo(f"   • Model files backed up: {models_backed_up}")
        click.echo(f"   • Backup location: {backup_dir}")
        
    except Exception as e:
        click.echo(f"Error creating backup: {e}")


@maintenance_group.command()
@click.option('--older-than-days', default=30, help='Remove files older than N days')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without actually deleting')
def cleanup(older_than_days, dry_run):
    """Clean up old model files and temporary data."""
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    
    action = "Would clean" if dry_run else "Cleaning"
    click.echo(f"{action} files older than {older_than_days} days (before {cutoff_date.strftime('%Y-%m-%d')})...")
    
    try:
        files_to_clean = []
        
        # Check models directory
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                file_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                if file_time < cutoff_date:
                    files_to_clean.append(model_file)
        
        # Check for temporary files
        temp_patterns = ["*.tmp", "temp_*", "*_temp.csv", "debug_*"]
        for pattern in temp_patterns:
            for temp_file in Path(".").glob(pattern):
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        files_to_clean.append(temp_file)
        
        if not files_to_clean:
            click.echo("No files found for cleanup.")
            return
        
        click.echo(f"Files to clean:")
        total_size = 0
        for file_path in files_to_clean:
            size = file_path.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            click.echo(f"  • {file_path} ({size_mb:.1f} MB)")
        
        click.echo(f"Total space to free: {total_size / (1024 * 1024):.1f} MB")
        
        if not dry_run:
            for file_path in files_to_clean:
                file_path.unlink()
            click.echo(f"✅ Cleanup completed! Removed {len(files_to_clean)} files.")
        else:
            click.echo(f"Dry run complete. Use without --dry-run to actually clean.")
        
    except Exception as e:
        click.echo(f"Error during cleanup: {e}")


@maintenance_group.command()
def validate():
    """Validate data integrity and model consistency."""
    click.echo("Validating NASCAR Fantasy Predictor system...")
    
    issues_found = []
    
    try:
        # Validate data integrity
        csv_manager = CSVDataManager()
        data_validation = csv_manager.validate_data_integrity()
        
        if data_validation['valid']:
            click.echo("✅ Data integrity: Valid")
        else:
            click.echo(f"❌ Data integrity: Issues found")
            for issue in data_validation['issues']:
                click.echo(f"   • {issue}")
                issues_found.append(f"Data: {issue}")
        
        # Validate model availability and consistency
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pth"))
            if model_files:
                # Try loading the latest model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                try:
                    predictor = NASCARPredictor(model_path=str(latest_model))
                    if predictor.trainer.model is not None:
                        click.echo("✅ Model loading: Success")
                    else:
                        click.echo("❌ Model loading: Failed to load model")
                        issues_found.append("Model: Failed to load latest model")
                except Exception as e:
                    click.echo(f"❌ Model loading: Error - {e}")
                    issues_found.append(f"Model: {e}")
            else:
                click.echo("⚠️ Model availability: No models found")
                issues_found.append("Model: No trained models available")
        else:
            click.echo("⚠️ Model directory: Not found")
            issues_found.append("Model: Models directory missing")
        
        # Summary
        if not issues_found:
            click.echo(f"\n✅ System validation passed! No issues found.")
        else:
            click.echo(f"\n⚠️ System validation found {len(issues_found)} issues:")
            for issue in issues_found:
                click.echo(f"   • {issue}")
            click.echo("\nRecommended actions:")
            click.echo("   • Run 'nascar-predictor data init' if data issues exist")
            click.echo("   • Run 'nascar-predictor train model' if model issues exist")
        
    except Exception as e:
        click.echo(f"Error during validation: {e}")


@maintenance_group.command()
@click.option('--output-file', default='system_info.json', help='Output file for system information')
def info(output_file):
    """Export detailed system information for debugging."""
    click.echo("Gathering system information...")
    
    try:
        import sys
        import torch
        
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'platform': sys.platform,
        }
        
        # Add data information
        try:
            csv_manager = CSVDataManager()
            stats = csv_manager.get_data_stats()
            system_info['data_status'] = stats
        except Exception as e:
            system_info['data_status'] = {'error': str(e)}
        
        # Add model information
        model_info = []
        model_dir = Path("models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.pth"):
                model_info.append({
                    'name': model_file.name,
                    'size_mb': model_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
        system_info['models'] = model_info
        
        # Save to file
        with open(output_file, 'w') as f:
            import json
            json.dump(system_info, f, indent=2)
        
        click.echo(f"✅ System information exported to {output_file}")
        click.echo(f"   • Python: {sys.version.split()[0]}")
        click.echo(f"   • PyTorch: {torch.__version__}")
        click.echo(f"   • Platform: {sys.platform}")
        
    except Exception as e:
        click.echo(f"Error gathering system info: {e}")