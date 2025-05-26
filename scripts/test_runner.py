#!/usr/bin/env python3
"""
Test runner script for MedVoiceQA tests.
Provides convenient commands for running different test suites.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd: list[str]) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="MedVoiceQA Test Runner")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "fast", "slow"],
        nargs="?",
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Run specific test file"
    )

    args = parser.parse_args()

    # Base pytest command
    cmd = ["pytest"]

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=pipeline", "--cov=nodes", "--cov=data", "--cov-report=html"])

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")

    # Add test type filtering
    if args.test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.test_type == "slow":
        cmd.extend(["-m", "slow"])

    # Add the specific file if provided
    if args.file:
        test_file = Path("tests") / args.file
        if not test_file.exists():
            print(f"Error: Test file {test_file} not found")
            return 1
        cmd.append(str(test_file))

    # Run the tests
    return run_command(cmd)


if __name__ == "__main__":
    sys.exit(main())
