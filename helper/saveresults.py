#!/usr/bin/env python3
"""
saveresults.py - Parse experiment results and append to structured CSV files.

This script reads raw terminal output from helper/results.txt, detects the result type
(training summary, retrieval evaluation, or reasoner metrics), prompts for metadata,
and appends parsed results to the appropriate CSV file.

Supported result types:
  1. Training summary (from train.py / wandb)  -> train_results.csv
     - Supports both "Run summary:" text format and JSON format
  2. Retrieval evaluation (from eval.py)       -> retrieval_results.csv
  3. Reasoner metrics (from reason/main.py)    -> reasoner_results.csv

Usage:
    cd helper
    python saveresults.py
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

RESULTS_TXT = Path(__file__).parent / "results.txt"
TRAIN_CSV = Path(__file__).parent / "train_results.csv"
RETRIEVAL_CSV = Path(__file__).parent / "retrieval_results.csv"
REASONER_CSV = Path(__file__).parent / "reasoner_results.csv"

# Column schemas for each CSV (ensures consistent column order)
TRAIN_COLUMNS = [
    "run_id", "dataset", "retriever_type", "gnn_num_layers", "gnn_hidden_dim",
    "epoch", "loss", "num_patient_epochs",
    "val_ans_recall_100", "val_triple_recall_100", "val_epoch",
    "runtime", "step", "timestamp", "wandb_runtime"
]

RETRIEVAL_COLUMNS = [
    "run_id", "dataset", "retriever_type", "gnn_num_layers", "gnn_hidden_dim",
    "K", "ans_recall", "shortest_path_triple_recall", "gpt_triple_recall"
]

REASONER_COLUMNS = [
    "run_id", "dataset", "retriever_type", "gnn_num_layers", "gnn_hidden_dim",
    "subset", "Hit", "Hit_at_1", "Macro_F1", "Macro_Precision", "Macro_Recall",
    "Exact_Match", "Totally_Wrong", "Hal_Score",
    "Micro_F1", "Micro_Precision", "Micro_Recall",
    "num_samples", "num_no_answer_samples", "no_answer_ratio"
]


# =============================================================================
# Parser Functions
# =============================================================================

def parse_train_summary(text: str) -> dict:
    """
    Parse a training summary block (typically from wandb Run summary).

    Supports two formats:

    1. Text format (with or without 'wandb:' prefix):
        Run summary:
        epoch 35
        loss 0.00679
        num_patient_epochs 10
        val/ans_recall@100 0.92327
        val/epoch 25
        val/triple_recall@100 0.87338

    2. JSON format:
        {
          "epoch": 44,
          "loss": 0.004614648803780447,
          "num_patient_epochs": 10,
          "val/ans_recall@100": 0.9528214397717172,
          "val/epoch": 34,
          "val/triple_recall@100": 0.8744843446836644,
          ...
        }

    Returns:
        dict with keys: epoch, loss, num_patient_epochs,
                        val_ans_recall_100, val_triple_recall_100, val_epoch,
                        runtime, step, timestamp, wandb_runtime
    """
    result: dict = {
        "epoch": None,
        "loss": None,
        "num_patient_epochs": None,
        "val_ans_recall_100": None,
        "val_triple_recall_100": None,
        "val_epoch": None,
        "runtime": None,
        "step": None,
        "timestamp": None,
        "wandb_runtime": None,
    }

    # Key mapping from raw text/JSON keys to our standardized column names
    key_mapping = {
        "epoch": "epoch",
        "loss": "loss",
        "num_patient_epochs": "num_patient_epochs",
        "val/ans_recall@100": "val_ans_recall_100",
        "val/triple_recall@100": "val_triple_recall_100",
        "val/epoch": "val_epoch",
        "_runtime": "runtime",
        "_step": "step",
        "_timestamp": "timestamp",
        "_wandb.runtime": "wandb_runtime",
    }

    # Try to parse as JSON first
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            data = json.loads(text_stripped)
            for json_key, col_name in key_mapping.items():
                if json_key in data:
                    value = data[json_key]
                    # Convert to appropriate type
                    if isinstance(value, float) and value.is_integer():
                        result[col_name] = int(value)
                    else:
                        result[col_name] = value
            return result
        except json.JSONDecodeError:
            pass  # Fall through to text parsing

    # Fall back to text parsing for "Run summary" format
    in_summary = False
    for line in text.splitlines():
        # Strip wandb: prefix if present
        line = re.sub(r"^wandb:\s*", "", line)
        line = line.strip()

        if "Run summary" in line:
            in_summary = True
            continue

        if in_summary:
            # Stop if we hit an empty line or a line that looks like a new section header
            if not line or line.startswith("=") or line.startswith("-"):
                break

            # Parse key-value pairs: expect "key value" format
            # Handle both space-separated and possible variations
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value_str = parts[1]

                if key in key_mapping:
                    col_name = key_mapping[key]
                    # Try to convert to appropriate numeric type
                    try:
                        if "." in value_str:
                            result[col_name] = float(value_str)
                        else:
                            result[col_name] = int(value_str)
                    except ValueError:
                        result[col_name] = value_str

    return result


def parse_retrieval_eval(text: str) -> list[dict]:
    """
    Parse a retrieval evaluation table from eval.py.

    Expected format:
        K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
        50       0.882                        0.814              0.784
        100       0.930                        0.871              0.844
        ...

    Returns:
        List of dicts, one per K value, with keys:
            K, ans_recall, shortest_path_triple_recall, gpt_triple_recall
    """
    results = []

    lines = text.splitlines()
    in_table = False

    for line in lines:
        line = line.strip()

        # Detect header line
        if re.match(r"^K\s+ans_recall", line):
            in_table = True
            continue

        if in_table:
            # Stop if we hit an empty line or non-numeric start
            if not line:
                break

            # Parse the row: K ans_recall shortest_path_triple_recall gpt_triple_recall
            parts = line.split()
            if len(parts) >= 4:
                try:
                    row = {
                        "K": int(parts[0]),
                        "ans_recall": float(parts[1]),
                        "shortest_path_triple_recall": float(parts[2]),
                        "gpt_triple_recall": float(parts[3]),
                    }
                    results.append(row)
                except ValueError:
                    # If parsing fails, stop (might be end of table)
                    break

    return results


def parse_reasoner_results(text: str) -> dict:
    """
    Parse reasoner metrics from reason/main.py output.

    Expected format:
        Evaluating on subset: False
        Results:
        ...
        Hit@1: 81.57, Macro F1: 70.77, Macro Precision: 78.12, ...
        Micro F1: 0.449, Micro precision: 0.772, Micro Recall: 0.316
        Total number of samples: 1628, no answer samples: 39, ratio: 0.023
        Hit: 86.79

    Returns:
        dict with all parsed metrics
    """
    result: dict = {
        "subset": None,
        "Hit": None,
        "Hit_at_1": None,
        "Macro_F1": None,
        "Macro_Precision": None,
        "Macro_Recall": None,
        "Exact_Match": None,
        "Totally_Wrong": None,
        "Hal_Score": None,
        "Micro_F1": None,
        "Micro_Precision": None,
        "Micro_Recall": None,
        "num_samples": None,
        "num_no_answer_samples": None,
        "no_answer_ratio": None,
    }

    # Parse subset flag
    subset_match = re.search(r"Evaluating on subset:\s*(True|False)", text, re.IGNORECASE)
    if subset_match:
        result["subset"] = subset_match.group(1).lower() == "true"
    else:
        result["subset"] = False  # Default to False if not found

    # Parse the Hit@1 line (contains multiple comma-separated key: value pairs)
    # Pattern: "Hit@1: 81.57..., Macro F1: 70.77..., ..."
    hit1_line_match = re.search(r"Hit@1:.*", text)
    if hit1_line_match:
        hit1_line = hit1_line_match.group(0)
        # Split by comma and parse each pair
        pairs = hit1_line.split(",")
        for pair in pairs:
            pair = pair.strip()
            if ":" in pair:
                key, value_str = pair.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()

                # Map keys to column names
                key_lower = key.lower().replace(" ", "_").replace("@", "_at_")
                if key_lower == "hit_at_1":
                    result["Hit_at_1"] = _safe_float(value_str)
                elif key_lower == "macro_f1":
                    result["Macro_F1"] = _safe_float(value_str)
                elif key_lower == "macro_precision":
                    result["Macro_Precision"] = _safe_float(value_str)
                elif key_lower == "macro_recall":
                    result["Macro_Recall"] = _safe_float(value_str)
                elif key_lower == "exact_match":
                    result["Exact_Match"] = _safe_float(value_str)
                elif key_lower == "totally_wrong":
                    result["Totally_Wrong"] = _safe_float(value_str)
                elif key_lower == "hal_score":
                    result["Hal_Score"] = _safe_float(value_str)

    # Parse Micro F1 line
    micro_line_match = re.search(r"Micro F1:.*", text, re.IGNORECASE)
    if micro_line_match:
        micro_line = micro_line_match.group(0)
        pairs = micro_line.split(",")
        for pair in pairs:
            pair = pair.strip()
            if ":" in pair:
                key, value_str = pair.split(":", 1)
                key = key.strip().lower()
                value_str = value_str.strip()

                if "micro f1" in key:
                    result["Micro_F1"] = _safe_float(value_str)
                elif "micro precision" in key:
                    result["Micro_Precision"] = _safe_float(value_str)
                elif "micro recall" in key:
                    result["Micro_Recall"] = _safe_float(value_str)

    # Parse "Total number of samples" line
    samples_match = re.search(
        r"Total number of samples:\s*(\d+).*?no answer samples:\s*(\d+).*?ratio:\s*([\d.]+)",
        text,
        re.IGNORECASE
    )
    if samples_match:
        result["num_samples"] = int(samples_match.group(1))
        result["num_no_answer_samples"] = int(samples_match.group(2))
        result["no_answer_ratio"] = float(samples_match.group(3))

    # Parse standalone "Hit:" line (overall hit, different from Hit@1)
    # Look for "Hit:" at the start of a line to avoid matching "Hit@1:"
    hit_match = re.search(r"^Hit:\s*([\d.]+)", text, re.MULTILINE)
    if hit_match:
        result["Hit"] = float(hit_match.group(1))

    return result


def _safe_float(s: str) -> Optional[float]:
    """Safely convert a string to float, returning None on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Result Type Detection
# =============================================================================

def detect_result_type(text: str) -> Optional[str]:
    """
    Detect which type of result the text contains.

    Returns:
        "train", "retrieval", "reasoner", or None if unknown
    """
    # Check for JSON format training summary (contains wandb keys like "val/ans_recall@100")
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            data = json.loads(text_stripped)
            if "val/ans_recall@100" in data or "epoch" in data:
                return "train"
        except json.JSONDecodeError:
            pass

    if "Run summary" in text:
        return "train"
    elif re.search(r"^K\s+ans_recall", text, re.MULTILINE):
        return "retrieval"
    elif "Hit@1:" in text:
        return "reasoner"
    else:
        return None


# =============================================================================
# Metadata Collection
# =============================================================================

def get_metadata() -> dict:
    """
    Prompt user for metadata about the experiment run.

    Returns:
        dict with run_id, dataset, retriever_type, gnn_num_layers, gnn_hidden_dim
    """
    print("\n--- Enter experiment metadata ---")

    run_id = input("run_id (e.g. 'webqsp_l2_seed1_gnn'): ").strip()
    if not run_id:
        run_id = "unnamed_run"

    dataset = input("dataset [default: webqsp]: ").strip()
    if not dataset:
        dataset = "webqsp"

    retriever_type = input("retriever_type (e.g. 'mlp', 'gnn') [default: '']: ").strip()

    gnn_num_layers_str = input("gnn_num_layers (int) [default: NaN]: ").strip()
    if gnn_num_layers_str:
        try:
            gnn_num_layers = int(gnn_num_layers_str)
        except ValueError:
            gnn_num_layers = None
    else:
        gnn_num_layers = None

    gnn_hidden_dim_str = input("gnn_hidden_dim (int) [default: NaN]: ").strip()
    if gnn_hidden_dim_str:
        try:
            gnn_hidden_dim = int(gnn_hidden_dim_str)
        except ValueError:
            gnn_hidden_dim = None
    else:
        gnn_hidden_dim = None

    return {
        "run_id": run_id,
        "dataset": dataset,
        "retriever_type": retriever_type,
        "gnn_num_layers": gnn_num_layers,
        "gnn_hidden_dim": gnn_hidden_dim,
    }


# =============================================================================
# CSV Handling
# =============================================================================

def append_to_csv(csv_path: Path, new_df: pd.DataFrame, columns: list[str]) -> None:
    """
    Append new rows to a CSV file, creating it if it doesn't exist.

    Args:
        csv_path: Path to the CSV file
        new_df: DataFrame with new rows to append
        columns: List of column names (ensures consistent column order)
    """
    # Ensure new_df has all columns (fill missing with NaN)
    for col in columns:
        if col not in new_df.columns:
            new_df[col] = None

    # Reorder columns to match schema
    new_df = new_df[columns]

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        # Ensure existing_df has all columns too
        for col in columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df = existing_df[columns]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main script entry point."""

    # 1. Read results.txt
    if not RESULTS_TXT.exists():
        print(f"Error: {RESULTS_TXT} does not exist.")
        print("Please paste your experiment results into that file first.")
        return

    text = RESULTS_TXT.read_text().strip()

    if not text:
        print(f"Error: {RESULTS_TXT} is empty.")
        print("Please paste your experiment results into that file first.")
        return

    # 2. Detect result type
    result_type = detect_result_type(text)

    if result_type is None:
        print("Error: Could not detect result type from the text.")
        print("Expected one of:")
        print("  - Training summary (contains 'Run summary')")
        print("  - Retrieval evaluation (contains 'K  ans_recall' header)")
        print("  - Reasoner metrics (contains 'Hit@1:')")
        return

    print(f"\nDetected result type: {result_type.upper()}")

    # 3. Get metadata from user
    metadata = get_metadata()

    # 4. Parse and append based on result type
    if result_type == "train":
        parsed = parse_train_summary(text)
        row = {**metadata, **parsed}
        new_df = pd.DataFrame([row])
        append_to_csv(TRAIN_CSV, new_df, TRAIN_COLUMNS)

        print(f"\n✓ Appended 1 training row to {TRAIN_CSV.name}")
        print(f"  run_id={metadata['run_id']}, epoch={parsed.get('epoch')}, "
              f"loss={parsed.get('loss')}, val_ans_recall_100={parsed.get('val_ans_recall_100')}")

    elif result_type == "retrieval":
        parsed_rows = parse_retrieval_eval(text)

        if not parsed_rows:
            print("Error: Could not parse any retrieval rows from the text.")
            return

        # Replicate metadata for each K row
        rows = [{**metadata, **row} for row in parsed_rows]
        new_df = pd.DataFrame(rows)
        append_to_csv(RETRIEVAL_CSV, new_df, RETRIEVAL_COLUMNS)

        k_values = [r["K"] for r in parsed_rows]
        print(f"\n✓ Appended {len(parsed_rows)} retrieval rows to {RETRIEVAL_CSV.name}")
        print(f"  run_id={metadata['run_id']}, K values: {k_values}")

    elif result_type == "reasoner":
        parsed = parse_reasoner_results(text)
        row = {**metadata, **parsed}
        new_df = pd.DataFrame([row])
        append_to_csv(REASONER_CSV, new_df, REASONER_COLUMNS)

        print(f"\n✓ Appended 1 reasoner row to {REASONER_CSV.name}")
        print(f"  run_id={metadata['run_id']}, subset={parsed.get('subset')}, "
              f"Hit@1={parsed.get('Hit_at_1')}, Macro_F1={parsed.get('Macro_F1')}")


if __name__ == "__main__":
    main()
