from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dota_ml.pipeline import run_base_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    result = run_base_pipeline(
        data_dir=config["data_dir"],
        model_params=config.get("model"),
        submission_path=config.get(
            "submission_path", "results/submissions/submission.csv"
        ),
    )

    print(f"Train shape: {result.X_all_train.shape}")
    print(f"Test shape: {result.X_all_test.shape}")
    print(f"Submission saved to: {result.submission_path}")


if __name__ == "__main__":
    main()

