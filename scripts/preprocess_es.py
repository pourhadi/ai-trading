#!/usr/bin/env python3
"""
Preprocesses ES futures CSV data to the format expected by the training scripts.

Reads an input CSV with columns like Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume,
and outputs a CSV with columns:
    timestamp,best_bid,best_ask,bid_size,ask_size,last_price

Usage:
    python scripts/preprocess_es.py --input es.csv --output historical_data.csv
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Preprocess ES CSV to standard training format.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="es.csv",
        help="Path to the raw ES CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="historical_data.csv",
        help="Path to the output processed CSV file"
    )
    args = parser.parse_args()

    # Load raw data
    df = pd.read_csv(args.input)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Combine Date and Time into a timestamp (seconds since epoch)
    if "Date" not in df.columns or "Time" not in df.columns:
        raise KeyError("Input CSV must contain 'Date' and 'Time' columns")
    dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                        format="%Y/%m/%d %H:%M:%S")
    # Convert to Unix timestamp in seconds
    df["timestamp"] = dt.astype("int64") / 1e9

    # Rename columns to match training script expectations
    rename_map = {
        "Low": "best_bid",
        "High": "best_ask",
        "BidVolume": "bid_size",
        "AskVolume": "ask_size",
        "Last": "last_price",
    }
    for src, dst in rename_map.items():
        if src not in df.columns:
            raise KeyError(f"Expected column '{src}' not found in input CSV")
    df = df.rename(columns=rename_map)

    # Select and reorder columns
    out_cols = ["timestamp", "best_bid", "best_ask", "bid_size", "ask_size", "last_price"]
    processed = df[out_cols].copy()

    # Save processed data
    processed.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()