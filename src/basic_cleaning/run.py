#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import wandb
import pandas as pd
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Ensure all steps use the same W&B project
    wb_project = os.getenv("WANDB_PROJECT", "build-ml-pipeline-for-short-term-rental-prices")

    run = wandb.init(project=wb_project, job_type="basic_cleaning")
    run.config.update(vars(args))

    logger.info(f"Using W&B project: {wb_project}")
    logger.info(f"Downloading artifact {args.input_artifact}")

    # Use the artifact and download it to a local directory
    artifact = run.use_artifact(args.input_artifact)
    artifact_dir = artifact.download()  # returns local folder path

    # Try to resolve a CSV file path inside the downloaded artifact dir
    artifact_basename = args.input_artifact.split(":")[0]
    candidate_paths = [
        Path(artifact_dir) / artifact_basename,
        Path(artifact_dir) / "sample.csv",
        Path(artifact_dir) / artifact_basename.replace(".csv", ""),
    ]

    csv_path = None
    for p in candidate_paths:
        if p.exists():
            csv_path = p
            break

    # Fallback: pick first csv in artifact_dir
    if csv_path is None:
        csv_files = list(Path(artifact_dir).glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]

    if csv_path is None:
        raise FileNotFoundError(f"No CSV file found inside downloaded artifact folder: {artifact_dir}")

    logger.info(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean the data: filter by price using provided args
    logger.info("Applying price filters")
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)].copy()

    # (Optional) drop rows with missing critical columns — keep it minimal
    # df = df.dropna(subset=["latitude", "longitude", "price"])

    # Save cleaned data
    output_file = "clean_sample.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned dataset to {output_file}")

    # Create and log artifact
    artifact_out = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact_out.add_file(output_file)
    run.log_artifact(artifact_out)
    logger.info(f"Uploaded cleaned artifact {args.output_artifact}")

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The W&B input artifact (e.g. sample.csv:latest or entity/project/artifact:version)",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact (e.g. clean_sample.csv)",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the cleaned output artifact (e.g. cleaned_data)",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A short description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum acceptable price — rows below this will be removed",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum acceptable price — rows above this will be removed",
        required=True,
    )

    args = parser.parse_args()

    go(args)
