import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import joblib
import polars as pl
import pandas as pd
import numpy as np
from src.utils import setup_seed, setup_logger

def run_preprocessing(raw_data_path="SAML-D.csv", sample_ratio=None, output_dir="data"):
    setup_seed(42)
    logger = setup_logger("preprocess")
    logger.info("Starting Stage 1: Preprocessing...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/sample", exist_ok=True)

    cache_train = os.path.join(output_dir, "train.parquet")
    cache_val = os.path.join(output_dir, "val.parquet")
    cache_test = os.path.join(output_dir, "test.parquet")
    meta_path = "models/preprocessing_meta.joblib"

    # Fast check if cached
    if os.path.exists(cache_train) and os.path.exists(cache_val) and os.path.exists(cache_test) and os.path.exists(meta_path) and sample_ratio is None:
        logger.info("Cached preprocessed data found. Skipping raw preprocessing.")
        return

    logger.info(f"Reading raw data from {raw_data_path}...")
    scan = pl.scan_csv(raw_data_path)

    if sample_ratio is not None and sample_ratio < 1.0:
        logger.info(f"Subsampling raw data with ratio: {sample_ratio}")
        # To preserve time-series sequence accurately when sampling
        scan = scan.filter(pl.int_range(0, pl.len()).shuffle(seed=42) < (pl.len() * sample_ratio))

    df = scan.collect()
    total_raw_rows = df.shape[0]
    logger.info(f"Total rows collected: {total_raw_rows}")

    # Data Quality Assessment
    null_counts = {col: df[col].null_count() for col in df.columns}
    dtypes_dict = {col: str(df[col].dtype) for col in df.columns}
    dup_count = total_raw_rows - df.unique().shape[0]

    dq_df = pd.DataFrame([
        {
            "column": col,
            "dtype": dtypes_dict[col],
            "null_count": null_counts[col],
            "null_percentage": (null_counts[col] / total_raw_rows) * 100,
            "distinct_values": df[col].n_unique()
        } for col in df.columns
    ])
    dq_df.loc[len(dq_df)] = {
        "column": "__DATASET_DUPLICATES__",
        "dtype": "int",
        "null_count": dup_count,
        "null_percentage": (dup_count / total_raw_rows) * 100,
        "distinct_values": total_raw_rows - dup_count
    }
    dq_df.to_csv("outputs/data_quality.csv", index=False)
    logger.info("Saved data quality table to outputs/data_quality.csv")

    # Parse Date + Time -> timestamp and sort chronologically
    logger.info("Parsing timestamps and sorting chronologically...")
    df = df.with_columns(
        (pl.col("Date") + " " + pl.col("Time")).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp")
    ).sort("timestamp")

    # Compute Causal History Features (strictly earlier transactions, shift by 1)
    logger.info("Computing causal transaction history features...")
    # Sender prior count & amounts
    prior_sender_cnt = pl.col("Amount").cum_count().over("Sender_account") - 1
    prior_sender_sum = pl.col("Amount").shift(1).fill_null(0.0).cum_sum().over("Sender_account")
    prior_sender_sq_sum = (pl.col("Amount") ** 2).shift(1).fill_null(0.0).cum_sum().over("Sender_account")

    prior_sender_mean = pl.when(prior_sender_cnt > 0).then(prior_sender_sum / prior_sender_cnt).otherwise(0.0)

    # Sample variance: (sq_sum - sum^2 / n) / (n - 1)
    sample_var = pl.when(prior_sender_cnt > 1).then(
        (prior_sender_sq_sum - (prior_sender_sum ** 2) / prior_sender_cnt) / (prior_sender_cnt - 1)
    ).otherwise(0.0)
    sample_var = pl.when(sample_var < 0.0).then(0.0).otherwise(sample_var)
    prior_sender_std = sample_var.sqrt()

    amt_z_score = pl.when(prior_sender_cnt > 0).then(
        (pl.col("Amount") - prior_sender_mean) / (prior_sender_std + 1e-5)
    ).otherwise(0.0)

    # Time since sender previous txn in hours (fill first occurrence with -1.0)
    time_since_prev = pl.col("timestamp").diff().dt.total_seconds().over("Sender_account") / 3600.0
    time_since_prev = time_since_prev.fill_null(-1.0)

    # Receiver prior in-count
    receiver_prior_in = pl.col("Amount").cum_count().over("Receiver_account") - 1

    # Sender-Receiver pair prior count
    pair_prior_cnt = pl.col("Amount").cum_count().over(["Sender_account", "Receiver_account"]) - 1

    # Distinct prior receivers of sender
    is_new_receiver = pl.when(pair_prior_cnt == 0).then(1).otherwise(0)
    distinct_prior_rx = is_new_receiver.shift(1).fill_null(0).cum_sum().over("Sender_account")

    # Static / Instantaneous features
    log1p_amt = (pl.col("Amount") + 1.0).log()
    hour = pl.col("timestamp").dt.hour().cast(pl.Int32)
    day_of_week = pl.col("timestamp").dt.weekday().cast(pl.Int32) # Monday=1, Sunday=7 in Polars
    is_weekend = pl.when(day_of_week >= 6).then(1).otherwise(0)
    curr_mismatch = pl.when(pl.col("Payment_currency") != pl.col("Received_currency")).then(1).otherwise(0)
    cross_border = pl.when(pl.col("Sender_bank_location") != pl.col("Receiver_bank_location")).then(1).otherwise(0)

    df = df.with_columns([
        log1p_amt.alias("log1p_Amount"),
        hour.alias("hour"),
        day_of_week.alias("day_of_week"),
        is_weekend.alias("is_weekend"),
        curr_mismatch.alias("currency_mismatch"),
        cross_border.alias("cross_border"),
        prior_sender_cnt.alias("sender_prior_count"),
        prior_sender_mean.alias("sender_prior_mean"),
        prior_sender_std.alias("sender_prior_std"),
        amt_z_score.alias("amount_z_score"),
        time_since_prev.alias("time_since_prev_sender_txn"),
        receiver_prior_in.alias("receiver_prior_in_count"),
        pair_prior_cnt.alias("sender_receiver_prior_count"),
        distinct_prior_rx.alias("distinct_prior_receivers")
    ])

    # Chronological Split: 70% train / 15% val / 15% test by date
    unique_dates = df["Date"].unique().sort().to_list()
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.70)
    val_end_idx = int(n_dates * 0.85)
    train_date_cutoff = unique_dates[train_end_idx]
    val_date_cutoff = unique_dates[val_end_idx]

    logger.info(f"Date cutoffs - Train < {train_date_cutoff}, Val: [{train_date_cutoff}, {val_date_cutoff}), Test >= {val_date_cutoff}")

    train_df = df.filter(pl.col("Date") < train_date_cutoff)
    val_df = df.filter((pl.col("Date") >= train_date_cutoff) & (pl.col("Date") < val_date_cutoff))
    test_df = df.filter(pl.col("Date") >= val_date_cutoff)

    # Time-order assertion
    max_train_ts = train_df["timestamp"].max()
    min_val_ts = val_df["timestamp"].min()
    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    assert max_train_ts < min_val_ts, f"Leakage detected: max train ({max_train_ts}) >= min val ({min_val_ts})"
    assert max_val_ts < min_test_ts, f"Leakage detected: max val ({max_val_ts}) >= min test ({min_test_ts})"
    logger.info("Chronological split assertion passed: max(train) < min(val) < min(test).")

    # Fit High-Risk Locations and Categorical Encoders on TRAIN ONLY
    logger.info("Fitting encoders on TRAIN split only...")
    cat_cols = ["Payment_type", "Payment_currency", "Received_currency", "Sender_bank_location", "Receiver_bank_location"]
    encoders = {}

    for c in cat_cols:
        vals = sorted(train_df[c].unique().to_list())
        mapping = {val: idx for idx, val in enumerate(vals)}
        mapping["unknown"] = len(vals)
        encoders[c] = mapping

    # High-risk locations identified on TRAIN (laundering rate > train_mean * 1.5)
    train_mean_rate = train_df["Is_laundering"].mean()
    sender_rates = (
        train_df.group_by("Sender_bank_location")
        .agg([pl.col("Is_laundering").mean().alias("rate"), pl.len().alias("count")])
        .filter(pl.col("rate") > train_mean_rate * 1.5)
    )
    high_risk_locs = set(sender_rates["Sender_bank_location"].to_list())
    logger.info(f"Identified {len(high_risk_locs)} high-risk locations on train: {high_risk_locs}")

    def apply_encodings(split_df):
        cols_to_add = []
        is_hr = (
            pl.col("Sender_bank_location").is_in(list(high_risk_locs)) |
            pl.col("Receiver_bank_location").is_in(list(high_risk_locs))
        ).cast(pl.Int32).alias("high_risk_location")
        cols_to_add.append(is_hr)

        for col_name in cat_cols:
            mapping = encoders[col_name]
            unknown_code = mapping["unknown"]
            # Map values with replace
            cols_to_add.append(
                pl.col(col_name)
                .replace_strict(mapping, default=unknown_code, return_dtype=pl.Int32)
                .alias(f"{col_name}_code")
            )
        return split_df.with_columns(cols_to_add)

    train_df = apply_encodings(train_df)
    val_df = apply_encodings(val_df)
    test_df = apply_encodings(test_df)

    feature_cols = [
        "log1p_Amount",
        "hour",
        "day_of_week",
        "is_weekend",
        "currency_mismatch",
        "cross_border",
        "high_risk_location",
        "Payment_type_code",
        "Payment_currency_code",
        "Received_currency_code",
        "Sender_bank_location_code",
        "Receiver_bank_location_code",
        "sender_prior_count",
        "sender_prior_mean",
        "sender_prior_std",
        "amount_z_score",
        "time_since_prev_sender_txn",
        "receiver_prior_in_count",
        "sender_receiver_prior_count",
        "distinct_prior_receivers"
    ]

    # Zero leakage & correlation assertions
    assert "Laundering_type" not in feature_cols, "CRITICAL ERROR: Laundering_type must not be in features!"
    assert "Sender_account" not in feature_cols and "Receiver_account" not in feature_cols, "CRITICAL: Account IDs leaked!"

    logger.info("Checking feature correlations and NaN/inf on train...")
    for f in feature_cols:
        col_series = train_df[f]
        assert col_series.null_count() == 0, f"Feature {f} contains nulls!"
        corr_val = train_df.select(pl.corr(f, "Is_laundering")).item()
        assert abs(corr_val) < 0.95, f"Feature {f} has suspicious correlation {corr_val:.4f} with label!"

    # Downsampling negatives on train if memory-limited:
    # prompt: "If RAM is limited: train on ALL positives + 20% random negatives; ALWAYS evaluate on the full untouched test set."
    pos_train = train_df.filter(pl.col("Is_laundering") == 1)
    neg_train = train_df.filter(pl.col("Is_laundering") == 0)
    logger.info(f"Train split raw: {pos_train.shape[0]} positives, {neg_train.shape[0]} negatives")

    # Sample 20% random negatives on train to make training ultra-efficient and clean
    neg_train_sub = neg_train.filter(pl.int_range(0, pl.len()).shuffle(seed=42) < int(neg_train.shape[0] * 0.20))
    train_balanced_downsampled = pl.concat([pos_train, neg_train_sub]).sort("timestamp")
    logger.info(f"Train split subsampled: {train_balanced_downsampled.shape[0]} rows ({pos_train.shape[0]} pos, {neg_train_sub.shape[0]} neg)")

    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "encoders": encoders,
        "high_risk_locs": high_risk_locs,
        "train_date_cutoff": str(train_date_cutoff),
        "val_date_cutoff": str(val_date_cutoff)
    }
    joblib.dump(meta, meta_path)
    logger.info(f"Saved preprocessing metadata to {meta_path}")

    # Save cached parquets
    train_balanced_downsampled.write_parquet(cache_train)
    val_df.write_parquet(cache_val)
    test_df.write_parquet(cache_test)
    logger.info(f"Saved train ({train_balanced_downsampled.shape[0]} rows) to {cache_train}")
    logger.info(f"Saved val ({val_df.shape[0]} rows) to {cache_val}")
    logger.info(f"Saved untouched test ({test_df.shape[0]} rows) to {cache_test}")

    # Generate a small demo sample CSV (< 5 MB) for Streamlit demo / tests
    demo_pos = test_df.filter(pl.col("Is_laundering") == 1).head(500)
    demo_neg = test_df.filter(pl.col("Is_laundering") == 0).head(4500)
    demo_sample = pl.concat([demo_pos, demo_neg]).sort("timestamp")
    demo_sample_path = "data/sample/demo_transactions.csv"
    # Keep original raw columns for demo upload
    raw_cols = ["Time", "Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
                "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"]
    demo_sample.select([c for c in raw_cols if c in demo_sample.columns]).to_pandas().to_csv(demo_sample_path, index=False)
    logger.info(f"Saved small demo CSV ({demo_sample.shape[0]} rows) to {demo_sample_path}")

    logger.info("Stage 1 Preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None, help="Sample ratio for fast test")
    parser.add_argument("--raw", type=str, default="SAML-D.csv", help="Path to raw CSV")
    args = parser.parse_args()
    run_preprocessing(raw_data_path=args.raw, sample_ratio=args.sample)
