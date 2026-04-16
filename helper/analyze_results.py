#!/usr/bin/env python3
"""
Analyze SubgraphRAG experiment results by GNN layer count.

This script loads train, retrieval, and reasoner results from CSV files,
groups them by the number of GNN layers (l0, l1, l2), and computes
mean ± std for each metric across seeds.

Usage:
    python helper/analyze_results.py
"""

import re
import sys
import pandas as pd


# ============================================================================
# Helper functions
# ============================================================================

def detect_layers_column(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Detect or derive a 'layers' column (integer: 0, 1, 2) from the dataframe.
    
    Priority:
    1. Look for integer column like 'gnn_num_layers', 'num_layers', 'layers'
    2. Look for string column containing 'l0', 'l1', 'l2' patterns
       (e.g., 'run_id', 'run_tag', 'config', 'name')
    
    Returns:
        DataFrame with a 'layers' column added.
    
    Raises:
        SystemExit if layers cannot be determined.
    """
    df = df.copy()
    
    # Priority 1: Check for integer column with layer info
    int_layer_cols = ['gnn_num_layers', 'num_layers', 'layers', 'n_layers', 'num_gnn_layers']
    for col in int_layer_cols:
        if col in df.columns:
            # Check if it has valid values
            if df[col].notna().any():
                df['layers'] = df[col].astype(int)
                print(f"[{df_name}] Using integer column '{col}' for layers.")
                return df
    
    # Priority 2: Look for string column with l0/l1/l2 pattern
    string_cols = ['run_id', 'run_tag', 'config', 'config_name', 'name', 'experiment']
    pattern = re.compile(r'l(\d+)')  # matches 'l0', 'l1', 'l2', etc.
    
    for col in string_cols:
        if col in df.columns and df[col].dtype == object:
            # Try to extract layer number from this column
            extracted = df[col].apply(lambda x: extract_layer_from_string(x, pattern))
            if extracted.notna().all():
                df['layers'] = extracted.astype(int)
                print(f"[{df_name}] Extracted layers from string column '{col}'.")
                return df
    
    # If we reach here, we couldn't find layers
    print(f"\n[ERROR] Could not find or derive 'layers' column in {df_name}.")
    print(f"Columns available: {list(df.columns)}")
    print("Expected either:")
    print("  - An integer column like 'gnn_num_layers'")
    print("  - A string column like 'run_id' containing 'l0', 'l1', 'l2'")
    sys.exit(1)


def extract_layer_from_string(s, pattern):
    """
    Extract layer number from a string like 'l2-1', 'gnn_l1_seed3', etc.
    Returns the integer or None if not found.
    """
    if pd.isna(s):
        return None
    match = pattern.search(str(s))
    if match:
        return int(match.group(1))
    return None


def get_metric_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Identify metric columns as numeric columns excluding identifiers.
    
    Excludes:
    - 'layers' (our grouping column)
    - Common identifiers: seed, run_id, step, timestamp, etc.
    - Any columns in the exclude_cols list
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Common identifier columns to exclude
    id_patterns = [
        'layers', 'seed', 'run_id', 'run_tag', 'config', 'name', 'experiment',
        'step', 'timestamp', 'wandb_runtime', 'runtime', 'epoch', 'val_epoch',
        'num_patient_epochs', 'dataset', 'retriever_type', 'gnn_num_layers',
        'gnn_hidden_dim', 'K', 'subset', 'num_samples', 'num_no_answer_samples'
    ]
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter out identifiers
    metric_cols = [
        col for col in numeric_cols
        if col.lower() not in [p.lower() for p in id_patterns]
        and col not in exclude_cols
    ]
    
    return metric_cols


def compute_summary(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    """
    Compute mean and std for each metric, grouped by layers.
    Returns a DataFrame with flattened column names like 'Hit@1_mean', 'Hit@1_std'.
    """
    if not metric_cols:
        print("  [WARNING] No metric columns found to aggregate.")
        return pd.DataFrame()
    
    grouped = df.groupby('layers')[metric_cols].agg(['mean', 'std'])
    
    # Flatten MultiIndex columns
    grouped.columns = [f'{metric}_{stat}' for metric, stat in grouped.columns]
    
    # Sort index to ensure 0, 1, 2 order
    grouped = grouped.sort_index()
    
    return grouped


def print_summary(summary: pd.DataFrame, title: str):
    """Pretty-print a summary table with a clear header."""
    print(f"\n{'='*70}")
    print(f"=== {title} ===")
    print('='*70)
    
    if summary.empty:
        print("  (No data available)")
        return
    
    # Configure pandas display options temporarily
    with pd.option_context(
        'display.max_columns', None,
        'display.width', None,
        'display.float_format', '{:.4f}'.format
    ):
        print(summary.to_string())
    print()


# ============================================================================
# Filtering functions for each CSV type
# ============================================================================

def filter_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter train_results.csv.
    Generally use all rows (one row per run).
    """
    print("\n[TRAIN] Applying filters...")
    print(f"  Original rows: {len(df)}")
    print(f"  No filters applied - using all rows.")
    return df


def filter_retrieval_df(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Prepare retrieval_results.csv for aggregation.
    Does NOT filter by K - we want to group by (layers, K).
    
    Returns:
        tuple: (dataframe, has_k_column)
    """
    print("\n[RETRIEVAL] Preparing data...")
    print(f"  Total rows: {len(df)}")
    
    # Check for K column (case-insensitive)
    k_col = None
    for col in df.columns:
        if col.lower() == 'k':
            k_col = col
            break
    
    if k_col:
        # Normalize column name to 'K' if needed
        if k_col != 'K':
            df = df.rename(columns={k_col: 'K'})
        unique_k = sorted(df['K'].unique())
        print(f"  Found 'K' column with values: {unique_k}")
        print(f"  Will group by (layers, K) - no filtering applied.")
        return df, True
    else:
        print("  No 'K' column found - will group by layers only.")
        return df, False


def filter_reasoner_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reasoner_results.csv.
    If subset column exists, restrict to full test set (subset=False or 'full').
    """
    print("\n[REASONER] Applying filters...")
    print(f"  Original rows: {len(df)}")
    
    # Check for subset column
    subset_cols = ['subset', 'subset_flag', 'is_subset']
    for col in subset_cols:
        if col in df.columns:
            # Check the type of values
            unique_vals = df[col].unique()
            print(f"  Found '{col}' column with values: {unique_vals}")
            
            # Filter for full test set
            if df[col].dtype == bool or set(unique_vals) <= {True, False}:
                df_filtered = df[df[col] == False].copy()
                print(f"  Filtered to subset=False: {len(df_filtered)} rows")
                return df_filtered
            elif 'full' in [str(v).lower() for v in unique_vals]:
                df_filtered = df[df[col].astype(str).str.lower() == 'full'].copy()
                print(f"  Filtered to subset='full': {len(df_filtered)} rows")
                return df_filtered
            else:
                # Unknown values, just filter out True-like values
                df_filtered = df[~df[col].astype(bool)].copy()
                print(f"  Filtered to falsy subset values: {len(df_filtered)} rows")
                return df_filtered
    
    print("  No subset column found - using all rows.")
    return df


# ============================================================================
# Main function
# ============================================================================

def main():
    """Main entry point for the analysis script."""
    
    print("="*70)
    print("SubgraphRAG Results Analysis - Grouped by GNN Layers")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. Load CSVs
    # -------------------------------------------------------------------------
    print("\n[LOADING CSVs]")
    
    try:
        train_df = pd.read_csv("helper/train_results.csv")
        print(f"  Loaded train_results.csv: {len(train_df)} rows, {len(train_df.columns)} columns")
    except FileNotFoundError:
        print("[ERROR] helper/train_results.csv not found.")
        sys.exit(1)
    
    try:
        retrieval_df = pd.read_csv("helper/retrieval_results.csv")
        print(f"  Loaded retrieval_results.csv: {len(retrieval_df)} rows, {len(retrieval_df.columns)} columns")
    except FileNotFoundError:
        print("[ERROR] helper/retrieval_results.csv not found.")
        sys.exit(1)
    
    try:
        reasoner_df = pd.read_csv("helper/reasoner_results.csv")
        print(f"  Loaded reasoner_results.csv: {len(reasoner_df)} rows, {len(reasoner_df.columns)} columns")
    except FileNotFoundError:
        print("[ERROR] helper/reasoner_results.csv not found.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 2. Detect/derive 'layers' column for each dataframe
    # -------------------------------------------------------------------------
    print("\n[DETECTING LAYERS COLUMN]")
    
    train_df = detect_layers_column(train_df, "train_results")
    retrieval_df = detect_layers_column(retrieval_df, "retrieval_results")
    reasoner_df = detect_layers_column(reasoner_df, "reasoner_results")
    
    # Print layer distribution
    print("\n[LAYER DISTRIBUTION]")
    print(f"  Train:     {dict(train_df['layers'].value_counts().sort_index())}")
    print(f"  Retrieval: {dict(retrieval_df['layers'].value_counts().sort_index())}")
    print(f"  Reasoner:  {dict(reasoner_df['layers'].value_counts().sort_index())}")
    
    # -------------------------------------------------------------------------
    # 3. Apply filters
    # -------------------------------------------------------------------------
    train_df_filtered = filter_train_df(train_df)
    retrieval_df_filtered, retrieval_has_k = filter_retrieval_df(retrieval_df)
    reasoner_df_filtered = filter_reasoner_df(reasoner_df)
    
    # -------------------------------------------------------------------------
    # 4. Identify metric columns and compute summaries
    # -------------------------------------------------------------------------
    print("\n[COMPUTING SUMMARIES]")
    
    # Train metrics
    train_metric_cols = get_metric_columns(train_df_filtered)
    print(f"  Train metrics: {train_metric_cols}")
    train_summary = compute_summary(train_df_filtered, train_metric_cols)
    
    # Retrieval metrics - group by (layers, K) if K exists, otherwise just layers
    retrieval_metric_cols = get_metric_columns(retrieval_df_filtered)
    print(f"  Retrieval metrics: {retrieval_metric_cols}")
    
    if retrieval_has_k:
        # Group by both layers and K
        retrieval_summary = retrieval_df_filtered.groupby(['layers', 'K'])[retrieval_metric_cols].agg(['mean', 'std'])
        retrieval_summary.columns = [f'{metric}_{stat}' for metric, stat in retrieval_summary.columns]
        retrieval_summary = retrieval_summary.sort_index()
        retrieval_output_file = "helper/retrieval_results_summary_by_layers_and_K.csv"
        retrieval_print_title = "RETRIEVAL RESULTS (grouped by layers and K)"
    else:
        # Fall back to grouping by layers only
        retrieval_summary = compute_summary(retrieval_df_filtered, retrieval_metric_cols)
        retrieval_output_file = "helper/retrieval_results_summary_by_layers.csv"
        retrieval_print_title = "RETRIEVAL RESULTS (grouped by layers; no K column found)"
    
    # Reasoner metrics
    reasoner_metric_cols = get_metric_columns(reasoner_df_filtered)
    print(f"  Reasoner metrics: {reasoner_metric_cols}")
    reasoner_summary = compute_summary(reasoner_df_filtered, reasoner_metric_cols)
    
    # -------------------------------------------------------------------------
    # 5. Save summaries to CSV
    # -------------------------------------------------------------------------
    print("\n[SAVING SUMMARIES]")
    
    train_summary.to_csv("helper/train_results_summary_by_layers.csv")
    print("  Saved: helper/train_results_summary_by_layers.csv")
    
    retrieval_summary.to_csv(retrieval_output_file)
    print(f"  Saved: {retrieval_output_file}")
    
    reasoner_summary.to_csv("helper/reasoner_results_summary_by_layers.csv")
    print("  Saved: helper/reasoner_results_summary_by_layers.csv")
    
    # -------------------------------------------------------------------------
    # 6. Pretty-print summaries
    # -------------------------------------------------------------------------
    print_summary(train_summary, "TRAIN RESULTS (grouped by layers)")
    print_summary(retrieval_summary, retrieval_print_title)
    print_summary(reasoner_summary, "REASONER RESULTS (full test, grouped by layers)")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
