from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple forecast horizons.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6], help="Forecast horizons in months.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for horizon in args.horizons:
        print(f"Training forecast horizon {horizon} month(s)...")
        subprocess.run([sys.executable, "-m", "src.train_forecast", "--horizon", str(horizon)], check=True)


if __name__ == "__main__":
    main()
