"""Test runner script for NASCAR Fantasy Predictor tests."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests and report results."""
    print("Running NASCAR Fantasy Predictor Tests")
    print("=" * 50)

    # Change to project root directory
    project_root = Path(__file__).parent.parent

    try:
        # Run pytest with verbose output
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--durations=10",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        print(f"\nTest run completed with exit code: {result.returncode}")

        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed.")

        return result.returncode

    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
