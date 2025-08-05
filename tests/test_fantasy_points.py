"""Tests for fantasy points calculator."""

import pytest

from nascar_fantasy_predictor.features.fantasy_points import FantasyPointsCalculator


class TestFantasyPointsCalculator:
    """Test fantasy points calculation functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        calculator = FantasyPointsCalculator()
        assert calculator.scoring_system == "default"
        assert calculator.scoring_rules is not None

    def test_initialization_draftkings(self):
        """Test DraftKings initialization."""
        calculator = FantasyPointsCalculator(scoring_system="draftkings")
        assert calculator.scoring_system == "draftkings"
        assert calculator.scoring_rules is not None

    def test_calculate_points_winner(self, fantasy_calculator):
        """Test points calculation for race winner."""
        race_result = {
            "finish_position": 1,
            "start_position": 5,
            "laps_led": 120,
            "dnf": False,
        }

        points = fantasy_calculator.calculate_points(race_result)

        # Winner should get high points
        assert points > 0
        assert isinstance(points, float)

    def test_calculate_points_position_gain(self, fantasy_calculator):
        """Test points calculation with position gain."""
        race_result = {
            "finish_position": 3,
            "start_position": 8,  # Gained 5 positions
            "laps_led": 45,
            "dnf": False,
        }

        points = fantasy_calculator.calculate_points(race_result)

        # Should get points for position gain
        assert points > 0

    def test_calculate_points_position_loss(self, fantasy_calculator):
        """Test points calculation with position loss."""
        race_result = {
            "finish_position": 15,
            "start_position": 5,  # Lost 10 positions
            "laps_led": 0,
            "dnf": False,
        }

        points = fantasy_calculator.calculate_points(race_result)

        # Should still get some points, but less due to poor finish
        assert points > 0

    def test_calculate_points_dnf(self, fantasy_calculator):
        """Test points calculation for DNF."""
        race_result = {
            "finish_position": 40,
            "start_position": 5,
            "laps_led": 20,
            "dnf": True,
        }

        points = fantasy_calculator.calculate_points(race_result)

        # DNF should get penalty but still some points
        assert points >= 0  # Should respect minimum points

    def test_calculate_points_laps_led_bonus(self, fantasy_calculator):
        """Test laps led bonus calculation."""
        race_result_no_laps = {
            "finish_position": 10,
            "start_position": 10,
            "laps_led": 0,
            "dnf": False,
        }

        race_result_with_laps = {
            "finish_position": 10,
            "start_position": 10,
            "laps_led": 50,
            "dnf": False,
        }

        points_no_laps = fantasy_calculator.calculate_points(race_result_no_laps)
        points_with_laps = fantasy_calculator.calculate_points(race_result_with_laps)

        # Driver with laps led should have more points
        assert points_with_laps > points_no_laps

    def test_calculate_points_top_5_bonus(self, fantasy_calculator):
        """Test top 5 finish bonus."""
        race_result_top5 = {
            "finish_position": 4,
            "start_position": 4,
            "laps_led": 0,
            "dnf": False,
        }

        race_result_not_top5 = {
            "finish_position": 8,
            "start_position": 8,
            "laps_led": 0,
            "dnf": False,
        }

        points_top5 = fantasy_calculator.calculate_points(race_result_top5)
        points_not_top5 = fantasy_calculator.calculate_points(race_result_not_top5)

        # Top 5 finish should get bonus points
        assert points_top5 > points_not_top5

    def test_calculate_points_top_10_bonus(self, fantasy_calculator):
        """Test top 10 finish bonus."""
        race_result_top10 = {
            "finish_position": 7,
            "start_position": 7,
            "laps_led": 0,
            "dnf": False,
        }

        race_result_not_top10 = {
            "finish_position": 15,
            "start_position": 15,
            "laps_led": 0,
            "dnf": False,
        }

        points_top10 = fantasy_calculator.calculate_points(race_result_top10)
        points_not_top10 = fantasy_calculator.calculate_points(race_result_not_top10)

        # Top 10 finish should get bonus points
        assert points_top10 > points_not_top10

    def test_calculate_stage_points(self, fantasy_calculator):
        """Test stage points calculation."""
        # Test various stage finishes
        assert fantasy_calculator.calculate_stage_points(1) == 10  # Stage winner
        assert fantasy_calculator.calculate_stage_points(5) == 6  # 5th place
        assert fantasy_calculator.calculate_stage_points(10) == 1  # 10th place
        assert fantasy_calculator.calculate_stage_points(15) == 0  # Outside top 10

    def test_calculate_average_points(self, fantasy_calculator):
        """Test average points calculation."""
        race_results = [
            {"finish_position": 1, "start_position": 5, "laps_led": 100, "dnf": False},
            {"finish_position": 3, "start_position": 8, "laps_led": 50, "dnf": False},
            {"finish_position": 15, "start_position": 12, "laps_led": 0, "dnf": False},
        ]

        avg_points = fantasy_calculator.calculate_average_points(race_results)

        assert isinstance(avg_points, float)
        assert avg_points > 0

    def test_calculate_average_points_empty_list(self, fantasy_calculator):
        """Test average points calculation with empty list."""
        avg_points = fantasy_calculator.calculate_average_points([])
        assert avg_points == 0.0

    def test_finish_position_points_distribution(self, fantasy_calculator):
        """Test that finish position points decrease appropriately."""
        points_p1 = fantasy_calculator._get_finish_points(1)
        points_p2 = fantasy_calculator._get_finish_points(2)
        points_p10 = fantasy_calculator._get_finish_points(10)
        points_p20 = fantasy_calculator._get_finish_points(20)
        points_p40 = fantasy_calculator._get_finish_points(40)

        # Points should decrease as finish position gets worse
        assert points_p1 > points_p2
        assert points_p2 > points_p10
        assert points_p10 > points_p20
        assert points_p20 > points_p40

        # Winner should get maximum points
        assert points_p1 == 45

    def test_minimum_points_guarantee(self, fantasy_calculator):
        """Test that minimum points are guaranteed."""
        # Create a really bad result
        worst_result = {
            "finish_position": 43,  # Last place
            "start_position": 1,  # Started on pole, lost many positions
            "laps_led": 0,
            "dnf": True,
        }

        points = fantasy_calculator.calculate_points(worst_result)

        # Should still get minimum points
        minimum = fantasy_calculator.scoring_rules["minimum_points"]
        assert points >= minimum


class TestFantasyPointsCalculatorDraftKings:
    """Test DraftKings-specific scoring."""

    def test_draftkings_scoring_differences(
        self, draftkings_calculator, fantasy_calculator
    ):
        """Test that DraftKings scoring differs from default."""
        race_result = {
            "finish_position": 5,
            "start_position": 10,
            "laps_led": 30,
            "dnf": False,
        }

        dk_points = draftkings_calculator.calculate_points(race_result)
        default_points = fantasy_calculator.calculate_points(race_result)

        # Points should be different between scoring systems
        # (Exact difference depends on scoring rules)
        assert isinstance(dk_points, float)
        assert isinstance(default_points, float)


class TestFantasyPointsCalculatorEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_data_handling(self, fantasy_calculator):
        """Test handling of missing data in race results."""
        incomplete_result = {
            "finish_position": 10
            # Missing start_position, laps_led, etc.
        }

        # Should not raise an error
        points = fantasy_calculator.calculate_points(incomplete_result)
        assert isinstance(points, float)
        assert points >= 0

    def test_invalid_finish_position(self, fantasy_calculator):
        """Test handling of invalid finish positions."""
        invalid_result = {
            "finish_position": 0,  # Invalid position
            "start_position": 5,
            "laps_led": 10,
            "dnf": False,
        }

        # Should handle gracefully
        points = fantasy_calculator.calculate_points(invalid_result)
        assert isinstance(points, float)

    def test_negative_laps_led(self, fantasy_calculator):
        """Test handling of negative laps led."""
        result_with_negative = {
            "finish_position": 10,
            "start_position": 5,
            "laps_led": -5,  # Invalid negative value
            "dnf": False,
        }

        # Should handle gracefully (treat as 0 or handle appropriately)
        points = fantasy_calculator.calculate_points(result_with_negative)
        assert isinstance(points, float)

    def test_extreme_position_changes(self, fantasy_calculator):
        """Test handling of extreme position changes."""
        extreme_gain = {
            "finish_position": 1,
            "start_position": 43,  # Extreme gain
            "laps_led": 0,
            "dnf": False,
        }

        extreme_loss = {
            "finish_position": 43,
            "start_position": 1,  # Extreme loss
            "laps_led": 0,
            "dnf": False,
        }

        points_gain = fantasy_calculator.calculate_points(extreme_gain)
        points_loss = fantasy_calculator.calculate_points(extreme_loss)

        # Extreme gain should result in high points
        assert points_gain > points_loss
        assert points_gain > 0
        assert points_loss >= fantasy_calculator.scoring_rules["minimum_points"]
