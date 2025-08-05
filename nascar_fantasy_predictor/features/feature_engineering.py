"""Simplified feature engineering for NASCAR data analysis."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.csv_manager import CSVDataManager


class FeatureEngineer:
    """Engineer features for NASCAR data analysis."""

    def __init__(self, data_manager: CSVDataManager):
        self.data_manager = data_manager

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add analytical features to race results dataframe."""
        df = df.copy()

        # Position-based features
        if "start_position" in df.columns and "finish_position" in df.columns:
            df["position_change"] = df["start_position"] - df["finish_position"]
            df["gained_positions"] = df["position_change"] > 0
            df["lost_positions"] = df["position_change"] < 0
            df["positions_gained"] = df["position_change"].clip(lower=0)
            df["positions_lost"] = (-df["position_change"]).clip(lower=0)

        # Performance categories
        if "finish_position" in df.columns:
            df["top5_finish"] = df["finish_position"] <= 5
            df["top10_finish"] = df["finish_position"] <= 10
            df["top20_finish"] = df["finish_position"] <= 20
            df["dnf"] = df["finish_position"] > 30  # Approximate DNF

        # Speed metrics (if available)
        speed_cols = ["green_flag_speed", "avg_last_10_speed", "total_speed"]
        for col in speed_cols:
            if col in df.columns:
                # Normalize speed within each race
                df[f"{col}_percentile"] = df.groupby("date")[col].rank(pct=True)

        # Track type encoding (simple categorical)
        if "track_type" in df.columns:
            track_types = {
                "Superspeedway": 1,
                "Intermediate": 2,
                "Short Track": 3,
                "Road Course": 4,
            }
            df["track_type_code"] = df["track_type"].map(track_types).fillna(0)

        return df

    def get_driver_summary_stats(
        self,
        driver_name: str,
        lookback_races: int = 10,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """Get summary statistics for a driver's recent performance."""
        recent_results = self.data_manager.get_driver_recent_results(
            driver_name, as_of_date, lookback_races
        )

        if recent_results.empty:
            return {}

        stats = {
            "driver_name": driver_name,
            "races_analyzed": len(recent_results),
            "avg_finish": recent_results["finish_position"].mean(),
            "best_finish": recent_results["finish_position"].min(),
            "worst_finish": recent_results["finish_position"].max(),
            "avg_start": (
                recent_results["start_position"].mean()
                if "start_position" in recent_results
                else None
            ),
            "top5_rate": (recent_results["finish_position"] <= 5).mean(),
            "top10_rate": (recent_results["finish_position"] <= 10).mean(),
            "dnf_rate": (recent_results["finish_position"] > 30).mean(),
            "consistency": recent_results["finish_position"].std(),
        }

        # Add position change stats if available
        if "start_position" in recent_results.columns:
            recent_results["position_change"] = (
                recent_results["start_position"] - recent_results["finish_position"]
            )
            stats["avg_position_change"] = recent_results["position_change"].mean()
            stats["best_gain"] = recent_results["position_change"].max()

        # Add speed stats if available
        if "green_flag_speed" in recent_results.columns:
            stats["avg_green_flag_speed"] = recent_results["green_flag_speed"].mean()
            stats["avg_speed_percentile"] = (
                recent_results.groupby("date")["green_flag_speed"].rank(pct=True).mean()
            )

        # Add track-specific stats
        track_stats = self._get_track_type_breakdown(recent_results)
        stats.update(track_stats)

        return stats

    def _get_track_type_breakdown(self, results: pd.DataFrame) -> Dict:
        """Get performance breakdown by track type."""
        if "track_type" not in results.columns:
            return {}

        track_stats = {}
        for track_type in results["track_type"].unique():
            if pd.notna(track_type):
                track_results = results[results["track_type"] == track_type]
                if len(track_results) > 0:
                    track_stats[
                        f'{track_type.lower().replace(" ", "_")}_avg_finish'
                    ] = track_results["finish_position"].mean()
                    track_stats[f'{track_type.lower().replace(" ", "_")}_races'] = len(
                        track_results
                    )

        return track_stats

    def get_matchup_analysis(
        self, driver1: str, driver2: str, lookback_races: int = 20
    ) -> pd.DataFrame:
        """Compare two drivers' head-to-head performance."""
        races = self.data_manager.list_races()

        matchup_data = []
        for race in races[-lookback_races:]:
            results = self.data_manager.get_race_results(race["date"])

            d1_result = results[results["driver_name"] == driver1]
            d2_result = results[results["driver_name"] == driver2]

            if not d1_result.empty and not d2_result.empty:
                matchup_data.append(
                    {
                        "date": race["date"],
                        "track_name": d1_result.iloc[0].get("track_name", "Unknown"),
                        f"{driver1}_finish": d1_result.iloc[0]["finish_position"],
                        f"{driver2}_finish": d2_result.iloc[0]["finish_position"],
                        f"{driver1}_won": d1_result.iloc[0]["finish_position"]
                        < d2_result.iloc[0]["finish_position"],
                    }
                )

        return pd.DataFrame(matchup_data)

    def get_track_history(self, track_name: str, num_races: int = 5) -> pd.DataFrame:
        """Get historical results for a specific track."""
        track_results = self.data_manager.get_track_history(track_name)

        if track_results.empty:
            return pd.DataFrame()

        # Get the most recent races
        races = track_results["date"].unique()
        recent_races = sorted(races)[-num_races:]

        return track_results[track_results["date"].isin(recent_races)]
