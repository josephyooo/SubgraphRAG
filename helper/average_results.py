#!/usr/bin/env python3
"""
average_results.py - Aggregate experiment results across multiple runs.

PURPOSE:
--------
This script reads experiment result CSVs (train, retrieval, reasoner) that may
contain multiple runs with different seeds or configurations, and computes the
mean and standard deviation of metric columns grouped by configuration columns.

HOW CONFIG VS METRIC COLUMNS ARE DETERMINED:
--------------------------------------------
Config columns (used for grouping):
  - Explicit config names: "dataset", "retriever_type", "gnn_num_layers",
    "gnn_hidden_dim", "gnn_dropout", "prompt_mode", "model_name", "K", "subset"
  - These define the experimental setup and are used for grouping runs.

Metric columns (aggregated with mean/std):
  - Train: columns containing "loss", "runtime", "epoch", "recall"
  - Retrieval: columns containing "ans_recall", "triple_recall", "shortest_path", "gpt_triple"
  - Reasoner: columns containing "Hit", "Macro", "Exact", "Hal", "Micro", "Wrong"

Ignored columns (bookkeeping, not used for grouping or aggregation):
  - "run_id", "run_tag", "seed", "run_idx", "timestamp", "wandb_runtime",
    "step", "num_samples", "num_no_answer_samples", "no_answer_ratio"

HOW TO RUN:
-----------
  # From repo root with default paths:
  python helper/average_results.py

  # With custom CSV paths:
  python helper/average_results.py \\
      --train_csv path/to/train.csv \\
      --retrieval_csv path/to/retrieval.csv \\
      --reasoner_csv path/to/reasoner.csv \\
      --out_dir path/to/output

OUTPUT:
-------
Aggregated CSVs are saved to:
  - <out_dir>/train_results_aggregated.csv
  - <out_dir>/retrieval_results_aggregated.csv
  - <out_dir>/reasoner_results_aggregated.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION: Tweak these lists to adjust column classification
# =============================================================================

# Columns explicitly treated as config (grouping) columns
CONFIG_COLUMNS = [
    "dataset",
    "retriever_type",
    "gnn_num_layers",
    "gnn_hidden_dim",
    "gnn_dropout",
    "prompt_mode",
    "model_name",
    "K",       # For retrieval results, K is part of the config
    "subset",  # For reasoner results, subset is part of the config
]

# Columns explicitly ignored (bookkeeping, per-run identifiers)
IGNORE_COLUMNS = [
    "run_id",
    "run_tag",
    "seed",
    "run_idx",
    "timestamp",
    "wandb_runtime",
    "step",
    "num_samples",
    "num_no_answer_samples",
    "no_answer_ratio",
    "num_patient_epochs",
]

# Metric detection patterns for each CSV type
# A column is a metric if its name contains any of these substrings (case-insensitive)
METRIC_PATTERNS = {
    "train": ["loss", "runtime", "epoch", "recall"],
    "retrieval": ["ans_recall", "triple_recall", "shortest_path", "gpt_triple"],
    "reasoner": ["hit", "macro", "exact", "hal", "micro", "wrong"],
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results (mean/std) across multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="helper/train_results.csv",
        help="Path to training results CSV.",
    )
    parser.add_argument(
        "--retrieval_csv",
        type=str,
        default="helper/retrieval_results.csv",
        help="Path to retrieval results CSV.",
    )
    parser.add_argument(
        "--reasoner_csv",
        type=str,
        default="helper/reasoner_results.csv",
        help="Path to reasoner results CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="helper/aggregated",
        help="Output directory for aggregated CSVs.",
    )
    return parser.parse_args()


def identify_columns(df: pd.DataFrame, csv_type: str) -> tuple[list[str], list[str]]:
    """
    Identify config and metric columns in a DataFrame.

    Args:
        df: The DataFrame to analyze.
        csv_type: One of "train", "retrieval", "reasoner".

    Returns:
        A tuple of (config_columns, metric_columns).
    """
    all_cols = set(df.columns)
    metric_patterns = METRIC_PATTERNS.get(csv_type, [])

    # Find config columns that exist in this DataFrame
    config_cols = [c for c in CONFIG_COLUMNS if c in all_cols]

    # Find metric columns based on patterns
    metric_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column matches any metric pattern
        if any(pattern.lower() in col_lower for pattern in metric_patterns):
            # Make sure it's numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)

    # Remove any overlap (config cols should not be metrics)
    metric_cols = [c for c in metric_cols if c not in config_cols]

    # Remove ignored columns from both lists
    config_cols = [c for c in config_cols if c not in IGNORE_COLUMNS]
    metric_cols = [c for c in metric_cols if c not in IGNORE_COLUMNS]

    return config_cols, metric_cols


def aggregate_results(
    df: pd.DataFrame, config_cols: list[str], metric_cols: list[str]
) -> pd.DataFrame:
    """
    Aggregate metrics by computing mean and std grouped by config columns.

    Args:
        df: The DataFrame with raw results.
        config_cols: Columns to group by.
        metric_cols: Columns to aggregate.

    Returns:
        A DataFrame with one row per unique config, containing:
        - All config columns
        - For each metric m: m_mean and m_std columns
    """
    if not config_cols:
        # If no config columns, treat entire DataFrame as one group
        agg_data = {}
        for m in metric_cols:
            agg_data[f"{m}_mean"] = [df[m].mean()]
            agg_data[f"{m}_std"] = [df[m].std()]
        return pd.DataFrame(agg_data)

    # Group by config columns and aggregate
    grouped = df.groupby(config_cols, dropna=False)

    agg_dict = {}
    for m in metric_cols:
        agg_dict[m] = ["mean", "std"]

    agg_df = grouped[metric_cols].agg(agg_dict)

    # Flatten multi-level column index
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]

    # Reset index to turn config columns back into regular columns
    agg_df = agg_df.reset_index()

    return agg_df


def process_csv(
    csv_path: str, csv_type: str, out_dir: str
) -> pd.DataFrame | None:
    """
    Process a single CSV file: load, identify columns, aggregate, and save.

    Args:
        csv_path: Path to the input CSV.
        csv_type: One of "train", "retrieval", "reasoner".
        out_dir: Output directory.

    Returns:
        The aggregated DataFrame, or None if processing failed.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"[WARNING] File not found: {csv_path} - skipping.")
        return None

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return None

    # Check if DataFrame has rows
    if df.empty:
        print(f"[WARNING] {csv_path} has no rows - skipping.")
        return None

    # Identify columns
    config_cols, metric_cols = identify_columns(df, csv_type)

    print(f"\n--- Processing: {csv_path} ---")
    print(f"  Config columns: {config_cols}")
    print(f"  Metric columns: {metric_cols}")

    if not metric_cols:
        print(f"[WARNING] No metric columns found in {csv_path} - skipping aggregation.")
        return None

    # Aggregate
    agg_df = aggregate_results(df, config_cols, metric_cols)

    # Save
    out_name = f"{csv_type}_results_aggregated.csv"
    out_path = os.path.join(out_dir, out_name)
    agg_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(agg_df)} rows)")

    return agg_df


def print_summary(df: pd.DataFrame | None, name: str, max_rows: int = 20):
    """Print a compact summary of an aggregated DataFrame."""
    header = f"==== {name.upper()} RESULTS (aggregated) ===="
    print(f"\n{header}")

    if df is None:
        print("  (No data)")
        return

    if len(df) > max_rows:
        print(f"  (Showing first {max_rows} of {len(df)} rows)")
        print(df.head(max_rows).to_string(index=False))
    else:
        print(df.to_string(index=False))


def main():
    args = parse_args()

    # Create output directory if needed
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Process each CSV
    train_agg = process_csv(args.train_csv, "train", out_dir)
    retrieval_agg = process_csv(args.retrieval_csv, "retrieval", out_dir)
    reasoner_agg = process_csv(args.reasoner_csv, "reasoner", out_dir)

    # Print summaries
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print_summary(train_agg, "TRAIN")
    print_summary(retrieval_agg, "RETRIEVAL")
    print_summary(reasoner_agg, "REASONER")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
