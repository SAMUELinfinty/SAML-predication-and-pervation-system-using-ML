import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score
from scipy.stats import rankdata
import lightgbm as lgb
from src.utils import setup_seed, setup_logger

def rank_normalize(scores):
    """Normalize arbitrary scores to [0, 1] using empirical ranks."""
    return rankdata(scores) / len(scores)

def run_model_training(train_path="data/train.parquet", val_path="data/val.parquet"):
    setup_seed(42)
    logger = setup_logger("models")
    logger.info("Starting Stage 3: Model Building...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    meta = joblib.load("models/preprocessing_meta.joblib")
    feature_cols = meta["feature_cols"]

    logger.info(f"Loading train ({train_path}) and val ({val_path})...")
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df["Is_laundering"].to_numpy()

    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df["Is_laundering"].to_numpy()

    pos_tr = int(y_train.sum())
    neg_tr = len(y_train) - pos_tr
    scale_pos = neg_tr / max(1, pos_tr)
    logger.info(f"Train size: {len(y_train)} (pos: {pos_tr}, neg: {neg_tr}, scale_pos_weight: {scale_pos:.2f})")
    logger.info(f"Val size: {len(y_val)} (pos: {int(y_val.sum())})")

    # 1. Baseline: Logistic Regression (balanced)
    logger.info("Training Baseline: Logistic Regression (balanced)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr_model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_val_scores = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_val_pr = average_precision_score(y_val, lr_val_scores)
    logger.info(f"Baseline Logistic Regression Val PR-AUC: {lr_val_pr:.4f}")

    # 2. M1 LightGBM
    logger.info("Training M1: LightGBM Classifier...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params_lgb = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": min(scale_pos, 50.0), # robust scale
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    lgb_model = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=callbacks
    )
    lgb_val_scores = lgb_model.predict(X_val)
    lgb_val_pr = average_precision_score(y_val, lgb_val_scores)
    logger.info(f"M1 LightGBM Best Iteration: {lgb_model.best_iteration}, Val PR-AUC: {lgb_val_pr:.4f}")

    # 3. M2 Random Forest (balanced_subsample, negatives downsampled 1:10 on train)
    logger.info("Training M2: Random Forest (balanced_subsample, negatives 1:10 downsampled)...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    np.random.seed(42)
    n_sample_neg = min(len(neg_idx), len(pos_idx) * 10)
    rf_neg_sample = np.random.choice(neg_idx, size=n_sample_neg, replace=False)
    rf_train_idx = np.concatenate([pos_idx, rf_neg_sample])
    np.random.shuffle(rf_train_idx)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train.iloc[rf_train_idx], y_train[rf_train_idx])
    rf_val_scores = rf_model.predict_proba(X_val)[:, 1]
    rf_val_pr = average_precision_score(y_val, rf_val_scores)
    logger.info(f"M2 Random Forest Val PR-AUC: {rf_val_pr:.4f}")

    # 4. M3 Isolation Forest (unsupervised)
    logger.info("Training M3: Isolation Forest (unsupervised)...")
    if_model = IsolationForest(
        n_estimators=100,
        max_samples=min(100000, len(X_train)),
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_train)
    # Higher score = riskier (invert scikit-learn anomaly score)
    if_val_scores = -if_model.score_samples(X_val)
    if_val_pr = average_precision_score(y_val, if_val_scores)
    logger.info(f"M3 Isolation Forest Val PR-AUC: {if_val_pr:.4f}")

    # 5. M4 Ensemble with Rank Normalization & Val PR-AUC Weight Optimization
    logger.info("Tuning M4 Ensemble weights on VALIDATION split...")
    rank_lgb = rank_normalize(lgb_val_scores)
    rank_rf = rank_normalize(rf_val_scores)
    rank_if = rank_normalize(if_val_scores)
    rank_lr = rank_normalize(lr_val_scores)

    # Grid search across weight combinations
    best_weights = [0.6, 0.3, 0.05, 0.05]
    best_val_pr = 0.0

    weight_candidates = [
        [0.70, 0.20, 0.05, 0.05],
        [0.60, 0.30, 0.05, 0.05],
        [0.50, 0.40, 0.05, 0.05],
        [0.40, 0.40, 0.10, 0.10],
        [0.80, 0.15, 0.025, 0.025],
        [0.55, 0.35, 0.05, 0.05]
    ]

    for w in weight_candidates:
        w_norm = np.array(w) / sum(w)
        combined = (
            w_norm[0] * rank_lgb +
            w_norm[1] * rank_rf +
            w_norm[2] * rank_if +
            w_norm[3] * rank_lr
        )
        pr = average_precision_score(y_val, combined)
        if pr > best_val_pr:
            best_val_pr = pr
            best_weights = w_norm.tolist()

    logger.info(f"Tuned Ensemble Weights (LGB, RF, IF, LR): {[round(x, 4) for x in best_weights]}, Val PR-AUC: {best_val_pr:.4f}")

    val_ens_scores = (
        best_weights[0] * rank_lgb +
        best_weights[1] * rank_rf +
        best_weights[2] * rank_if +
        best_weights[3] * rank_lr
    )

    # 6. Calibrate Ensemble with Isotonic Regression on VALIDATION
    logger.info("Calibrating ensemble with Isotonic Regression on VALIDATION...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_ens_scores, y_val)
    val_calibrated_probs = calibrator.predict(val_ens_scores)

    # 7. Pick Decision Threshold on VALIDATION ONLY
    logger.info("Optimizing decision threshold on VALIDATION split...")
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_calibrated_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_val_f1 = float(f1_scores[best_idx])
    logger.info(f"Optimal Threshold chosen on VALIDATION: {best_threshold:.4f} (Val F1: {best_val_f1:.4f})")

    # Hyperparameter Summary Table
    hp_data = [
        {"model": "Logistic Regression (Baseline)", "hyperparameters": "penalty=l2, class_weight=balanced, max_iter=500", "val_pr_auc": f"{lr_val_pr:.4f}"},
        {"model": "LightGBM (M1)", "hyperparameters": f"num_leaves=31, max_depth=7, lr=0.05, scale_pos_weight={min(scale_pos, 50.0):.1f}, n_est={lgb_model.best_iteration}", "val_pr_auc": f"{lgb_val_pr:.4f}"},
        {"model": "Random Forest (M2)", "hyperparameters": "n_estimators=100, max_depth=12, class_weight=balanced_subsample, neg_sample=1:10", "val_pr_auc": f"{rf_val_pr:.4f}"},
        {"model": "Isolation Forest (M3)", "hyperparameters": "n_estimators=100, contamination=0.01, max_samples=100000", "val_pr_auc": f"{if_val_pr:.4f}"},
        {"model": "Ensemble (M4)", "hyperparameters": f"weights={dict(zip(['LGB','RF','IF','LR'], [round(x,3) for x in best_weights]))}, isotonic_calibrated", "val_pr_auc": f"{best_val_pr:.4f}"}
    ]
    hp_df = pd.DataFrame(hp_data)
    hp_df.to_csv("models/hyperparameters.csv", index=False)
    hp_df.to_csv("outputs/hyperparameters.csv", index=False)
    logger.info("Saved hyperparameter table to models/hyperparameters.csv and outputs/hyperparameters.csv")

    # Save complete bundle
    bundle = {
        "models": {
            "lightgbm": lgb_model,
            "random_forest": rf_model,
            "isolation_forest": if_model,
            "logistic_regression": lr_model,
            "scaler": scaler
        },
        "weights": best_weights,
        "calibrator": calibrator,
        "threshold": best_threshold,
        "encoders": meta["encoders"],
        "cat_cols": meta["cat_cols"],
        "high_risk_locs": meta["high_risk_locs"],
        "feature_cols": feature_cols
    }
    joblib.dump(bundle, "models/ensemble.joblib")
    logger.info("Saved complete ensemble bundle to models/ensemble.joblib")
    logger.info("Stage 3 Model Building completed successfully.")

if __name__ == "__main__":
    run_model_training()
