"""Fantasy points calculation for different scoring systems."""

from typing import Dict, Optional


class FantasyPointsCalculator:
    """Calculate fantasy points based on race results."""

    def __init__(self, scoring_system: str = "default"):
        self.scoring_system = scoring_system
        self.scoring_rules = self._get_scoring_rules(scoring_system)

    def calculate_points(self, race_result: Dict) -> float:
        """Calculate fantasy points for a single race result."""
        points = 0.0

        # Base points by finish position
        finish_pos = race_result.get("finish_position", 43)
        points += self._get_finish_points(finish_pos)

        # Laps led bonus
        laps_led = race_result.get("laps_led", 0)
        points += laps_led * self.scoring_rules["points_per_lap_led"]

        # Start position vs finish position (overtake points)
        start_pos = race_result.get("start_position")
        if start_pos and finish_pos:
            positions_gained = start_pos - finish_pos
            if positions_gained > 0:
                points += (
                    positions_gained * self.scoring_rules["points_per_position_gained"]
                )

        # Win bonus
        if finish_pos == 1:
            points += self.scoring_rules["win_bonus"]

        # Top 5 bonus
        if finish_pos <= 5:
            points += self.scoring_rules["top_5_bonus"]

        # Top 10 bonus
        if finish_pos <= 10:
            points += self.scoring_rules["top_10_bonus"]

        # DNF penalty
        if race_result.get("dnf", False):
            points += self.scoring_rules["dnf_penalty"]  # This is negative

        # Minimum points guarantee
        points = max(points, self.scoring_rules["minimum_points"])

        return round(points, 2)

    def _get_finish_points(self, finish_position: int) -> float:
        """Get base points for finish position."""
        if finish_position == 1:
            return 45
        elif finish_position == 2:
            return 42
        elif finish_position == 3:
            return 40
        elif finish_position <= 5:
            return 38 - (finish_position - 4)
        elif finish_position <= 10:
            return 35 - (finish_position - 6)
        elif finish_position <= 15:
            return 29 - (finish_position - 11)
        elif finish_position <= 20:
            return 24 - (finish_position - 16)
        elif finish_position <= 25:
            return 19 - (finish_position - 21)
        elif finish_position <= 30:
            return 14 - (finish_position - 26)
        elif finish_position <= 35:
            return 9 - (finish_position - 31)
        else:
            return max(1, 4 - (finish_position - 36))

    def _get_scoring_rules(self, system: str) -> Dict:
        """Get scoring rules for different fantasy systems."""
        systems = {
            "default": {
                "points_per_lap_led": 0.25,
                "points_per_position_gained": 0.5,
                "win_bonus": 5,
                "top_5_bonus": 2,
                "top_10_bonus": 1,
                "dnf_penalty": -5,
                "minimum_points": 1,
            },
            "draftkings": {
                "points_per_lap_led": 0.25,
                "points_per_position_gained": 0.5,
                "win_bonus": 5,
                "top_5_bonus": 3,
                "top_10_bonus": 1,
                "dnf_penalty": -5,
                "minimum_points": 0,
            },
            "fanduel": {
                "points_per_lap_led": 0.1,
                "points_per_position_gained": 0.5,
                "win_bonus": 10,
                "top_5_bonus": 3,
                "top_10_bonus": 1,
                "dnf_penalty": -2,
                "minimum_points": 1,
            },
        }

        return systems.get(system, systems["default"])

    def calculate_stage_points(self, stage_finish: int) -> float:
        """Calculate fantasy points for stage finishes."""
        if stage_finish == 1:
            return 10
        elif stage_finish == 2:
            return 9
        elif stage_finish == 3:
            return 8
        elif stage_finish <= 10:
            return 11 - stage_finish
        else:
            return 0

    def calculate_average_points(self, race_results: list) -> float:
        """Calculate average fantasy points across multiple races."""
        if not race_results:
            return 0.0

        total_points = sum(self.calculate_points(result) for result in race_results)
        return round(total_points / len(race_results), 2)
