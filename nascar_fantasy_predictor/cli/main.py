"""Simple NASCAR data collection and analysis CLI."""

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from ..data.csv_manager import CSVDataManager
from ..data.lapraptor_scraper import LapRaptorScraper
from ..data.nascar_official_scraper import NASCAROfficialScraper
from ..data.qualifying_scraper import QualifyingScraper
from ..features.fantasy_points import FantasyPointsCalculator
from ..features.feature_engineering import FeatureEngineer


@click.group()
@click.version_option(version="0.1.0", prog_name="nascar-data")
def main():
    """NASCAR Data Collection and Analysis Tool."""
    pass


@main.command()
@click.option(
    "--year", default=2025, help="Year to collect (currently only 2025 supported)"
)
def collect(year):
    """Collect NASCAR race data."""
    manager = CSVDataManager()

    if year != 2025:
        click.echo("Currently only 2025 data collection is supported")
        return

    click.echo(f"Collecting NASCAR data for {year}...")

    # Collect from LapRaptor
    lapraptor = LapRaptorScraper()

    try:
        # Get list of completed races
        races = lapraptor.get_completed_2025_races()
        click.echo(f"Found {len(races)} completed races in 2025")

        collected_count = 0
        for race_info in races:
            race_id = race_info["race_id"]
            race_name = race_info["race_name"]

            click.echo(f"Fetching {race_name}...")

            # Get race results
            results = lapraptor.get_race_results(race_id)
            if results is not None and not results.empty:
                # Get loop data for additional metrics
                loop_data = lapraptor.get_loop_data(race_id)

                # Merge data if loop data exists
                if loop_data is not None and not loop_data.empty:
                    results = results.merge(
                        loop_data[
                            [
                                "driver_name",
                                "green_flag_speed",
                                "avg_last_10_speed",
                                "total_speed",
                            ]
                        ],
                        on="driver_name",
                        how="left",
                    )

                # Save to CSV
                manager.save_race_results(results, race_info["date"], race_name)
                collected_count += 1
                click.echo(f"✓ Saved {race_name}")
            else:
                click.echo(f"✗ No results for {race_name}")

        click.echo(f"\nData collection complete! Collected {collected_count} races.")

    except Exception as e:
        click.echo(f"✗ Error collecting data: {e}")
        import traceback

        traceback.print_exc()


@main.command()
@click.option("--date", help='Race date (YYYY-MM-DD) or "latest" for most recent')
@click.option("--track", help="Track name")
def analyze(date, track):
    """Analyze race data and generate insights."""
    manager = CSVDataManager()

    # Load race data
    if date == "latest":
        # Get all data and find the latest race
        all_data = manager.load_race_data()
        if all_data.empty:
            click.echo("No race data found. Run 'collect' first.")
            return
        date = all_data["date"].max()
        click.echo(f"Using latest race date: {date}")

    if date:
        results = manager.get_race_results(date)
    elif track:
        results = manager.get_track_history(track)
    else:
        click.echo("Please specify --date or --track")
        return

    if results.empty:
        click.echo("No data found for specified criteria")
        return

    # Calculate fantasy points
    calculator = FantasyPointsCalculator(scoring_system="draftkings")
    results["dk_points"] = results.apply(
        lambda row: calculator.calculate_points(row.to_dict()), axis=1
    )

    # Display insights
    click.echo(f"\nRace Analysis for {date or track}")
    click.echo("=" * 60)

    # Top performers
    click.echo("\nTop 10 Fantasy Performers:")
    top_10 = results.nlargest(10, "dk_points")[
        ["driver_name", "finish_position", "start_position", "dk_points"]
    ]
    for idx, row in top_10.iterrows():
        click.echo(
            f"{row['driver_name']:20} | Finish: P{int(row['finish_position']):2d} | Start: P{int(row['start_position']):2d} | DK Points: {row['dk_points']:.1f}"
        )

    # Position changes
    results["position_change"] = results["start_position"] - results["finish_position"]

    click.echo("\nBiggest Movers:")
    movers = results.nlargest(5, "position_change")[
        ["driver_name", "position_change", "start_position", "finish_position"]
    ]
    for idx, row in movers.iterrows():
        click.echo(
            f"{row['driver_name']:20} | +{int(row['position_change'])} positions (P{int(row['start_position'])} → P{int(row['finish_position'])})"
        )


@main.command()
@click.option("--driver", required=True, help="Driver name")
@click.option("--track", help="Track name (optional)")
@click.option("--year", type=int, help="Year (optional)")
def driver_stats(driver, track, year):
    """Get detailed statistics for a specific driver."""
    manager = CSVDataManager()

    # Get driver history
    df = manager.get_driver_data(driver_name=driver)

    if df.empty:
        click.echo(f"No data found for driver: {driver}")
        return

    # Filter by track/year if specified
    if track:
        df = df[df["track_name"].str.contains(track, case=False, na=False)]
    if year:
        df = df[pd.to_datetime(df["date"]).dt.year == year]

    if df.empty:
        click.echo("No data found for specified criteria")
        return

    # Calculate stats
    click.echo(f"\nDriver Statistics: {driver}")
    click.echo("=" * 60)
    click.echo(f"Races: {len(df)}")
    click.echo(f"Average Finish: {df['finish_position'].mean():.1f}")
    click.echo(f"Best Finish: P{int(df['finish_position'].min())}")
    click.echo(f"Worst Finish: P{int(df['finish_position'].max())}")
    click.echo(f"Top 5s: {(df['finish_position'] <= 5).sum()}")
    click.echo(f"Top 10s: {(df['finish_position'] <= 10).sum()}")

    if "laps_led" in df.columns:
        click.echo(f"Total Laps Led: {df['laps_led'].sum()}")

    # Recent form
    recent = df.nlargest(5, "date")[["date", "track_name", "finish_position"]]
    click.echo("\nRecent Results:")
    for idx, row in recent.iterrows():
        click.echo(
            f"{row['date']} | {row['track_name'][:20]:20} | P{int(row['finish_position'])}"
        )


@main.command()
@click.option(
    "--format", default="csv", type=click.Choice(["csv", "json"]), help="Export format"
)
@click.option("--output", required=True, help="Output filename")
@click.option(
    "--year", type=int, help="Year to export (optional, exports all if not specified)"
)
def export(format, output, year):
    """Export collected data for external analysis."""
    manager = CSVDataManager()

    # Load all data or specific year
    if year:
        # Load data for specific year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        df = manager.load_race_data(start_date=start_date, end_date=end_date)
    else:
        # Load all data
        df = manager.load_race_data()

    if df.empty:
        click.echo("No data to export")
        return

    # Add feature engineering
    engineer = FeatureEngineer(manager)
    df = engineer.create_features(df)

    # Export
    if format == "csv":
        df.to_csv(output, index=False)
    else:
        df.to_json(output, orient="records", indent=2)

    click.echo(f"Exported {len(df)} race results to {output}")


if __name__ == "__main__":
    main()
