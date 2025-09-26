#!/usr/bin/env python3
"""
Backtesting script for AI Financial Fraud Detection System.

This module provides functionality to run the trained ensemble model over
a historical transactions dataset between a given date range and reports
key performance metrics for fraud detection.

Usage:
    python scripts/backtest.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
        --data-path data/processed/transactions.csv --model-path models/ensemble_model

Example:
    python scripts/backtest.py --start-date 2023-01-01 --end-date 2023-12-31

Notes:
    - This is a minimal implementation to satisfy README references.
    - Replace data loading paths and column names with your project schema.
    - Ensure the dataset contains 'is_fraud' label column and timestamp column.

Author: AI Financial Fraud Detection Team
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Make src importable when running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from models.ensemble_model import FraudDetectionEnsemble  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the backtest script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Backtest fraud detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start-date 2023-01-01 --end-date 2023-12-31
  %(prog)s --start-date 2023-01-01 --end-date 2023-12-31 \
           --data-path custom/path/transactions.csv
        """
    )
    
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--data-path",
        default="data/processed/transactions.csv",
        help="Path to CSV file with historical transactions including is_fraud label"
    )
    
    parser.add_argument(
        "--model-path",
        default="models/ensemble_model",
        help="Directory containing saved ensemble model artifacts"
    )
    
    return parser.parse_args()


def load_data(path: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Load and filter transaction data for the specified date range.
    
    Args:
        path (str): Path to the CSV file containing transaction data.
        start (datetime): Start date for filtering.
        end (datetime): End date for filtering.
        
    Returns:
        pd.DataFrame: Filtered transaction data.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
        
    df = pd.read_csv(path)
    
    # Expect columns: timestamp, is_fraud, and feature columns used by the model
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # Parse timestamps
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        print("Warning: No timestamp column found. Using all available data.")
    
    return df


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> Dict[str, Any]:
    """
    Calculate performance metrics for fraud detection model.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        y_proba (pd.Series): Predicted probabilities.
        
    Returns:
        Dict[str, Any]: Dictionary containing calculated metrics.
    """
    metrics = {
        "n_samples": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    return metrics


def print_results(metrics: Dict[str, Any]) -> None:
    """
    Print backtest results in a formatted manner.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing calculated metrics.
    """
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    
    print(f"Dataset Statistics:")
    print(f"  Sample Count: {metrics['n_samples']:,}")
    print(f"  Fraud Rate: {metrics['fraud_rate']:.4f} ({metrics['fraud_rate']*100:.2f}%)")
    
    print(f"\nModel Performance:")
    print(f"  AUC-ROC Score: {metrics['auc_roc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print("=" * 50)


def main() -> int:
    """
    Main function to execute the backtesting process.
    
    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        args = parse_args()
        start = datetime.fromisoformat(args.start_date)
        end = datetime.fromisoformat(args.end_date)
        
        print(f"Loading data from {args.data_path} between {start} and {end}...")
        data = load_data(args.data_path, start, end)
        
        if data.empty:
            print("No data found in the specified date range.")
            return 0
            
        if "is_fraud" not in data.columns:
            print("Error: dataset must contain 'is_fraud' label column.")
            return 1
            
        # Separate features and label; drop non-feature identifiers if present
        y = data["is_fraud"].astype(int)
        columns_to_drop = [c for c in ["is_fraud", "transaction_id", "timestamp"] 
                          if c in data.columns]
        X = data.drop(columns=columns_to_drop)
        
        print(f"Loading model from {args.model_path}...")
        model = FraudDetectionEnsemble.load(args.model_path)
        
        print("Running backtest predictions...")
        proba = model.predict_proba(X)
        preds = model.predict(X)
        
        # Calculate and display metrics
        metrics = calculate_metrics(y, preds, proba)
        print_results(metrics)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
