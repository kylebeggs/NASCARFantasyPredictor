"""Main CLI entry point for NASCAR Fantasy Predictor."""

import click
from .commands.data_commands import data_group
from .commands.training_commands import training_group
from .commands.prediction_commands import prediction_group
from .commands.analysis_commands import analysis_group
from .commands.maintenance_commands import maintenance_group


@click.group()
@click.version_option()
def cli():
    """NASCAR Fantasy Predictor - AI-powered fantasy league picks."""
    pass


# Add command groups
cli.add_command(data_group)
cli.add_command(training_group)
cli.add_command(prediction_group)
cli.add_command(analysis_group)
cli.add_command(maintenance_group)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()