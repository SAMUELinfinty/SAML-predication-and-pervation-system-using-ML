import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import rankdata
import shap
import lightgbm as lgb
from src.utils import setup_seed, setup_logger

def rank_normalize(scores):
    return rankdata(scores) / len(scores)

def get_english_reason(feature, val, shap_val):
    feat_names_readable = {
        "log1p_Amount": f"Amount is unusually high (log-value: {val:.2f})",
        "amount_z_score": f"Amount deviates significantly ({val:+.2f} std dev) from sender historical average",
        "cross_border": "Cross-border payment between different country jurisdictions",
        "currency_mismatch": "Currency exchange conversion mismatch between sender and receiver",
        "high_risk_location": "Involves designated high-risk offshore or foreign banking jurisdiction",
        "distinct_prior_receivers": f"Sender has dispersed funds across {int(val)} distinct receiver accounts",
        "time_since_prev_sender_txn": f"Abnormal transaction interval ({val:.1f} hours since previous activity)",
        "sender_prior_count": f"Sender velocity anomaly ({int(val)} prior transactions recorded)",
        "receiver_prior_in_count": f"Receiver account velocity anomaly ({int(val)} prior incoming transfers)",
        "sender_receiver_prior_count": f"First-time or rare transfer pair (prior co-occurrence: {int(val)})",
        "Payment_type_code": "High-risk payment execution channel (Cash/Wire/Cross-border)",
        "hour": f"Off-hours transaction timing (hour {int(val)}:00)",
        "day_of_week": f"Day of week pattern (day {int(val)})",
        "is_weekend": "Weekend execution window",
        "sender_prior_mean": f"Historical average sender amount (${val:,.2f})",
        "sender_prior_std": f"Historical sender amount volatility (std ${val:,.2f})",
        "Payment_currency_code": "Foreign originating currency code",
        "Received_currency_code": "Foreign destination currency code",
        "Sender_bank_location_code": "Originating bank jurisdiction code",
        "Receiver_bank_location_code": "Destination bank jurisdiction code"
    }
    desc = feat_names_readable.get(feature, f"Feature '{feature}' value {val:.2f}")
    return f"{desc} (SHAP impact: {shap_val:+.3f})"

def run_evaluation(test_path="data/test.parquet", models_dir="models", outputs_dir="outputs"):
    setup_seed(42)
    logger = setup_logger("evaluate")
    logger.info("Starting Stage 4: Result Analysis & Stress Tests...")
    fig_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    bundle = joblib.load(os.path.join(models_dir, "ensemble.joblib"))
    models = bundle["models"]
    weights = bundle["weights"]
    calibrator = bundle["calibrator"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]

    logger.info(f"Loading untouched TEST split from {test_path}...")
    test_df = pl.read_parquet(test_path)
    y_test = test_df["Is_laundering"].to_numpy()
    X_test = test_df.select(feature_cols).to_pandas()
    n_test = len(y_test)
    pos_test = int(y_test.sum())
    logger.info(f"Untouched TEST size: {n_test:,} (positives: {pos_test:,}, rate: {pos_test/n_test*100:.4f}%)")

    # Score each component model
    logger.info("Scoring component models on TEST...")
    # Baseline LR
    X_test_scaled = models["scaler"].transform(X_test)
    lr_scores = models["logistic_regression"].predict_proba(X_test_scaled)[:, 1]

    # M1 LightGBM
    lgb_scores = models["lightgbm"].predict(X_test)

    # M2 Random Forest
    rf_scores = models["random_forest"].predict_proba(X_test)[:, 1]

    # M3 Isolation Forest
    if_scores = -models["isolation_forest"].score_samples(X_test)

    # M4 Ensemble
    logger.info("Computing M4 Ensemble scores and calibration...")
    rank_lgb = rank_normalize(lgb_scores)
    rank_rf = rank_normalize(rf_scores)
    rank_if = rank_normalize(if_scores)
    rank_lr = rank_normalize(lr_scores)

    ens_raw = (
        weights[0] * rank_lgb +
        weights[1] * rank_rf +
        weights[2] * rank_if +
        weights[3] * rank_lr
    )
    calibrated_probs = calibrator.predict(ens_raw)
    preds = (calibrated_probs >= threshold).astype(int)

    # Check Quality Gate 6: test PR-AUC > 0.99 check
    test_pr_auc = average_precision_score(y_test, calibrated_probs)
    lgb_pr_auc = average_precision_score(y_test, lgb_scores)
    logger.info(f"Test PR-AUC - Ensemble: {test_pr_auc:.4f}, LightGBM: {lgb_pr_auc:.4f}")
    assert test_pr_auc <= 0.99, f"Quality Gate 6 Violation: Test PR-AUC {test_pr_auc:.4f} > 0.99 indicates potential target leakage!"

    # Calculate Core Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    spec = tn / (tn + fp)
    fpr = fp / (tn + fp)
    roc_auc = roc_auc_score(y_test, calibrated_probs)
    mcc = matthews_corrcoef(y_test, preds)

    # Precision@K and Recall@K (K = 0.1%, 0.5%, 1.0%)
    sorted_order = np.argsort(-calibrated_probs)
    sorted_y = y_test[sorted_order]
    pk_results = {}
    for k_pct in [0.001, 0.005, 0.010]:
        k_count = int(n_test * k_pct)
        tp_at_k = sorted_y[:k_count].sum()
        p_at_k = tp_at_k / k_count
        r_at_k = tp_at_k / pos_test
        pk_results[f"precision_at_{k_pct*100:.1f}pct"] = p_at_k
        pk_results[f"recall_at_{k_pct*100:.1f}pct"] = r_at_k
        logger.info(f"Top {k_pct*100:.1f}% ({k_count:,} reviews): Precision={p_at_k:.4f}, Recall={r_at_k:.4f} (TPs caught: {int(tp_at_k)}/{pos_test})")

    # Per-Typology Recall
    typology_series = test_df["Laundering_type"].to_pandas()
    test_pred_series = pd.Series(preds, index=test_df.to_pandas().index)
    pos_mask = (y_test == 1)
    typology_recalls = {}
    for typ in typology_series[pos_mask].value_counts().head(8).index:
        typ_pos_idx = np.where((y_test == 1) & (typology_series == typ))[0]
        typ_rec = preds[typ_pos_idx].sum() / max(1, len(typ_pos_idx))
        typology_recalls[f"recall_typology_{typ}"] = typ_rec

    # Per-Payment_Type Recall
    pt_series = test_df["Payment_type"].to_pandas()
    pt_recalls = {}
    for pt in pt_series[pos_mask].unique():
        pt_pos_idx = np.where((y_test == 1) & (pt_series == pt))[0]
        pt_rec = preds[pt_pos_idx].sum() / max(1, len(pt_pos_idx))
        pt_recalls[f"recall_payment_{pt}"] = pt_rec

    # Temporal Drift: First-half vs Second-half of TEST
    half_split = n_test // 2
    pr_auc_h1 = average_precision_score(y_test[:half_split], calibrated_probs[:half_split])
    pr_auc_h2 = average_precision_score(y_test[half_split:], calibrated_probs[half_split:])
    rec_h1 = recall_score(y_test[:half_split], preds[:half_split], zero_division=0)
    rec_h2 = recall_score(y_test[half_split:], preds[half_split:], zero_division=0)
    logger.info(f"Drift Analysis - Test H1 PR-AUC: {pr_auc_h1:.4f}, H2 PR-AUC: {pr_auc_h2:.4f} | H1 Recall: {rec_h1:.4f}, H2 Recall: {rec_h2:.4f}")

    # Compile Comprehensive Metrics Table
    metrics_dict = {
        "model": "Ensemble (M4)",
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "pr_auc": test_pr_auc,
        "mcc": mcc,
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "test_total": n_test,
        "test_positives": pos_test,
        "precision_at_0.1pct": pk_results["precision_at_0.1pct"],
        "recall_at_0.1pct": pk_results["recall_at_0.1pct"],
        "precision_at_0.5pct": pk_results["precision_at_0.5pct"],
        "recall_at_0.5pct": pk_results["recall_at_0.5pct"],
        "precision_at_1.0pct": pk_results["precision_at_1.0pct"],
        "recall_at_1.0pct": pk_results["recall_at_1.0pct"],
        "drift_h1_pr_auc": pr_auc_h1,
        "drift_h2_pr_auc": pr_auc_h2,
        "drift_h1_recall": rec_h1,
        "drift_h2_recall": rec_h2
    }
    metrics_dict.update(typology_recalls)
    metrics_dict.update(pt_recalls)

    # Component Models metrics row
    comp_models_metrics = [
        {"model": "Logistic Regression", "pr_auc": average_precision_score(y_test, lr_scores), "roc_auc": roc_auc_score(y_test, lr_scores)},
        {"model": "LightGBM", "pr_auc": lgb_pr_auc, "roc_auc": roc_auc_score(y_test, lgb_scores)},
        {"model": "Random Forest", "pr_auc": average_precision_score(y_test, rf_scores), "roc_auc": roc_auc_score(y_test, rf_scores)},
        {"model": "Isolation Forest", "pr_auc": average_precision_score(y_test, if_scores), "roc_auc": roc_auc_score(y_test, if_scores)},
        {"model": "Ensemble (M4)", "pr_auc": test_pr_auc, "roc_auc": roc_auc}
    ]
    pd.DataFrame(comp_models_metrics).to_csv(os.path.join(outputs_dir, "model_comparison.csv"), index=False)

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(outputs_dir, "metrics.csv"), index=False)
    logger.info("Saved metrics to outputs/metrics.csv")

    # Generate Publication Plots
    # 1. PR Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sc in [("Logistic Regression", lr_scores), ("Random Forest", rf_scores), ("Isolation Forest", if_scores), ("LightGBM", lgb_scores), ("Ensemble", calibrated_probs)]:
        p_c, r_c, _ = precision_recall_curve(y_test, sc)
        ax.plot(r_c, p_c, label=f"{name} (PR-AUC = {average_precision_score(y_test, sc):.3f})", linewidth=2.0)
    ax.set_title("Precision-Recall Curves on Untouched TEST Split", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sc in [("Logistic Regression", lr_scores), ("Random Forest", rf_scores), ("Isolation Forest", if_scores), ("LightGBM", lgb_scores), ("Ensemble", calibrated_probs)]:
        fpr_c, tpr_c, _ = roc_curve(y_test, sc)
        ax.plot(fpr_c, tpr_c, label=f"{name} (ROC-AUC = {roc_auc_score(y_test, sc):.3f})", linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Predicted Legit", "Predicted Laundering"], yticklabels=["Actual Legit", "Actual Laundering"])
    ax.set_title(f"Ensemble Confusion Matrix (Threshold={threshold:.3f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 4. Model Comparison Bar Chart
    comp_df = pd.DataFrame(comp_models_metrics)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df["pr_auc"], width, label="PR-AUC", color="#2980b9")
    ax.bar(x + width/2, comp_df["roc_auc"], width, label="ROC-AUC", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["model"], rotation=20, ha="right")
    ax.set_title("Detector Benchmark: PR-AUC vs ROC-AUC on TEST", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison_bars.png"), dpi=300)
    plt.close()

    # 5. Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fop_raw, mpv_raw = calibration_curve(y_test, ens_raw, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)
    ax.plot(mpv_raw, fop_raw, marker="o", label="Uncalibrated Ensemble", color="#e74c3c")
    ax.plot(mpv_cal, fop_cal, marker="s", label="Isotonic Calibrated Ensemble", color="#2ecc71")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "calibration_curve.png"), dpi=300)
    plt.close()

    # SHAP Explainability on LightGBM
    logger.info("Computing SHAP explanations...")
    pos_idx_test = np.where(y_test == 1)[0]
    neg_idx_test = np.where(y_test == 0)[0]
    np.random.seed(42)
    sample_pos = np.random.choice(pos_idx_test, size=min(3000, len(pos_idx_test)), replace=False)
    sample_neg = np.random.choice(neg_idx_test, size=min(7000, len(neg_idx_test)), replace=False)
    shap_idx = np.concatenate([sample_pos, sample_neg])
    X_shap = X_test.iloc[shap_idx]
    y_shap = y_test[shap_idx]

    explainer = shap.TreeExplainer(models["lightgbm"])
    shap_values = explainer.shap_values(X_shap)
    # Binary classification tree explainer may return list or array
    if isinstance(shap_values, list):
        shap_vals_arr = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals_arr = shap_values[:, :, 1]
    else:
        shap_vals_arr = shap_values

    # Global SHAP Bar Plot
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals_arr, X_shap, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_summary_bar.png"), dpi=300)
    plt.close()

    # SHAP Beeswarm Plot
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals_arr, X_shap, show=False)
    plt.title("SHAP Beeswarm Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_beeswarm.png"), dpi=300)
    plt.close()

    # Save Mean |SHAP| Importance CSV
    mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
    shap_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_imp_df.to_csv(os.path.join(outputs_dir, "shap_importance.csv"), index=False)

    # SHAP Waterfall for a True Positive (TP) and False Positive (FP)
    flagged_in_shap = np.where((preds[shap_idx] == 1))[0]
    tp_in_shap = np.where((preds[shap_idx] == 1) & (y_shap == 1))[0]
    fp_in_shap = np.where((preds[shap_idx] == 1) & (y_shap == 0))[0]

    base_val = explainer.expected_value
    if isinstance(base_val, list):
        base_val = base_val[1]

    if len(tp_in_shap) > 0:
        tp_i = tp_in_shap[0]
        tp_exp = shap.Explanation(
            values=shap_vals_arr[tp_i],
            base_values=base_val,
            data=X_shap.iloc[tp_i].values,
            feature_names=feature_cols
        )
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(tp_exp, max_display=8, show=False)
        plt.title("SHAP Waterfall: True Positive Case", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "shap_waterfall_tp.png"), dpi=300)
        plt.close()

    if len(fp_in_shap) > 0:
        fp_i = fp_in_shap[0]
        fp_exp = shap.Explanation(
            values=shap_vals_arr[fp_i],
            base_values=base_val,
            data=X_shap.iloc[fp_i].values,
            feature_names=feature_cols
        )
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(fp_exp, max_display=8, show=False)
        plt.title("SHAP Waterfall: False Positive Case", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "shap_waterfall_fp.png"), dpi=300)
        plt.close()

    # Per-flagged transaction top-3 plain English reasons (cap 50k rows)
    logger.info("Generating top-3 plain English explanations for flagged transactions...")
    flagged_indices = np.where(preds == 1)[0]
    cap_flagged = flagged_indices[:50000]
    X_flagged = X_test.iloc[cap_flagged]

    # Batch compute shap values for flagged transactions
    flagged_shaps = explainer.shap_values(X_flagged)
    if isinstance(flagged_shaps, list):
        flagged_shaps = flagged_shaps[1]
    elif flagged_shaps.ndim == 3:
        flagged_shaps = flagged_shaps[:, :, 1]

    flagged_reasons = []
    amounts_flagged = test_df["Amount"].to_numpy()[cap_flagged]
    dates_flagged = test_df["Date"].to_numpy()[cap_flagged]
    times_flagged = test_df["Time"].to_numpy()[cap_flagged]

    for idx, orig_i in enumerate(range(len(cap_flagged))):
        shaps_row = flagged_shaps[orig_i]
        top3_f_idx = np.argsort(-shaps_row)[:3]
        row_vals = X_flagged.iloc[orig_i]
        r1 = get_english_reason(feature_cols[top3_f_idx[0]], row_vals.iloc[top3_f_idx[0]], shaps_row[top3_f_idx[0]])
        r2 = get_english_reason(feature_cols[top3_f_idx[1]], row_vals.iloc[top3_f_idx[1]], shaps_row[top3_f_idx[1]])
        r3 = get_english_reason(feature_cols[top3_f_idx[2]], row_vals.iloc[top3_f_idx[2]], shaps_row[top3_f_idx[2]])
        flagged_reasons.append({
            "test_index": cap_flagged[orig_i],
            "Date": str(dates_flagged[orig_i]),
            "Time": str(times_flagged[orig_i]),
            "Amount": amounts_flagged[orig_i],
            "calibrated_prob": round(float(calibrated_probs[cap_flagged[orig_i]]), 4),
            "actual_label": int(y_test[cap_flagged[orig_i]]),
            "reason_1": r1,
            "reason_2": r2,
            "reason_3": r3
        })
    flagged_df = pd.DataFrame(flagged_reasons)
    flagged_df.to_csv(os.path.join(outputs_dir, "flagged_with_reasons.csv"), index=False)
    logger.info(f"Saved {len(flagged_df)} flagged transactions with plain English reasons to outputs/flagged_with_reasons.csv")

    # STRESS TESTS
    # (a) Imbalance Stress Test: 1:1, 10:1, 100:1, natural
    logger.info("Executing Stress Test A: Imbalance ratios (1:1, 10:1, 100:1, natural)...")
    imbalance_results = []
    ratios = [("1:1", 1), ("10:1", 10), ("100:1", 100), ("Natural", len(neg_idx_test) / pos_test)]
    for r_name, r_val in ratios:
        n_neg = min(len(neg_idx_test), int(pos_test * r_val))
        sub_neg = np.random.choice(neg_idx_test, size=n_neg, replace=False)
        sub_idx = np.concatenate([pos_idx_test, sub_neg])
        sub_y = y_test[sub_idx]
        sub_probs = calibrated_probs[sub_idx]
        sub_pr = average_precision_score(sub_y, sub_probs)
        sub_roc = roc_auc_score(sub_y, sub_probs)
        imbalance_results.append({
            "ratio_name": r_name,
            "neg_pos_ratio": r_val,
            "test_sample_size": len(sub_idx),
            "positives": pos_test,
            "negatives": n_neg,
            "pr_auc": sub_pr,
            "roc_auc": sub_roc
        })
        logger.info(f"Imbalance {r_name} -> PR-AUC: {sub_pr:.4f}, ROC-AUC: {sub_roc:.4f}")
    pd.DataFrame(imbalance_results).to_csv(os.path.join(outputs_dir, "stress_test_imbalance.csv"), index=False)

    # (b) Black-box Adversarial Evasion Stress Test
    logger.info("Executing Stress Test B: Adversarial Evasion & Defense...")
    # Attacker controllable features: Amount (and log1p_Amount), Payment_type_code, hour
    # Perturb test positives to attempt evasion
    clean_pos_preds = preds[pos_idx_test]
    baseline_clean_recall = clean_pos_preds.mean()

    # Adversarial perturbation: scale amount by 0.9, 0.8, 0.7, 0.5 & random search 50 variations
    scales = [0.9, 0.8, 0.7, 0.5]
    X_test_pos = X_test.iloc[pos_idx_test].copy()
    perturbed_best = X_test_pos.copy()
    lowest_scores = calibrated_probs[pos_idx_test].copy()

    for p_i in range(len(pos_idx_test)):
        orig_row = X_test_pos.iloc[p_i].copy()
        best_row = orig_row.copy()
        min_prob = lowest_scores[p_i]

        # 50 perturbations search
        for _ in range(50):
            cand = orig_row.copy()
            # Random scale among amounts
            s = np.random.choice(scales)
            raw_amt = np.expm1(cand["log1p_Amount"]) * s
            cand["log1p_Amount"] = np.log1p(raw_amt)
            cand["amount_z_score"] = cand["amount_z_score"] * s
            # Shift hour by +/- 2
            cand["hour"] = int((cand["hour"] + np.random.choice([-2, -1, 1, 2])) % 24)

            # Evaluate LightGBM score
            cand_df = pd.DataFrame([cand])
            cand_lgb_score = models["lightgbm"].predict(cand_df)[0]
            if cand_lgb_score < min_prob:
                min_prob = cand_lgb_score
                best_row = cand

        perturbed_best.iloc[p_i] = best_row

    # Score perturbed positives with current ensemble
    pert_lgb_scores = models["lightgbm"].predict(perturbed_best)
    # Approximate rank score of perturbed positives
    pert_preds = (pert_lgb_scores >= threshold).astype(int)
    adversarial_evaded_recall = pert_preds.mean()
    logger.info(f"Adversarial Evasion: Clean Positive Recall = {baseline_clean_recall:.4f} -> Under Attack Recall = {adversarial_evaded_recall:.4f}")

    # Adversarial Defense: Adversarial Retraining (add perturbed positives to train)
    logger.info("Adversarial Retraining: adding perturbed samples to training set...")
    train_df = pl.read_parquet("data/train.parquet")
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df["Is_laundering"].to_numpy()

    # Append perturbed positives to train
    X_adv_train = pd.concat([X_train, perturbed_best], ignore_index=True)
    y_adv_train = np.concatenate([y_train, np.ones(len(perturbed_best), dtype=int)])

    lgb_adv = lgb.train(
        {
            "objective": "binary",
            "metric": "average_precision",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "verbosity": -1,
            "random_state": 42
        },
        lgb.Dataset(X_adv_train, label=y_adv_train),
        num_boost_round=100
    )

    # Evaluate retrained model
    retrained_clean_pr_auc = average_precision_score(y_test, lgb_adv.predict(X_test))
    recovered_pos_preds = (lgb_adv.predict(perturbed_best) >= threshold).astype(int)
    recovered_recall = recovered_pos_preds.mean()
    logger.info(f"Adversarial Defense Result: Recovered Recall Under Attack = {recovered_recall:.4f}, Clean PR-AUC = {retrained_clean_pr_auc:.4f}")

    adv_results = [
        {"metric": "Clean Positive Recall (Pre-Attack)", "value": f"{baseline_clean_recall:.4f}"},
        {"metric": "Adversarial Attack Recall (Evasion Drop)", "value": f"{adversarial_evaded_recall:.4f}"},
        {"metric": "Adversarially Retrained Recall (Defense Recovery)", "value": f"{recovered_recall:.4f}"},
        {"metric": "Clean Test PR-AUC Post-Defense Retraining", "value": f"{retrained_clean_pr_auc:.4f}"}
    ]
    pd.DataFrame(adv_results).to_csv(os.path.join(outputs_dir, "adversarial_results.csv"), index=False)

    logger.info("Stage 4 Result Analysis, SHAP, and Stress Tests completed successfully.")

if __name__ == "__main__":
    run_evaluation()
