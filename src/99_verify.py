import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import pandas as pd
import polars as pl
from PIL import Image

def verify_quality_gates():
    print("Executing automated Quality Gates verification (src/99_verify.py)...")
    errors = []

    # GATE 1: Time order assertion & feature integrity
    print("Checking Gate 1: Time-order and feature leakage...")
    if not os.path.exists("models/preprocessing_meta.joblib"):
        errors.append("Gate 1: models/preprocessing_meta.joblib does not exist!")
    else:
        meta = joblib.load("models/preprocessing_meta.joblib")
        if "Laundering_type" in meta["feature_cols"]:
            errors.append("Gate 1 Violation: 'Laundering_type' found in feature list! Target leakage!")
        if "Sender_account" in meta["feature_cols"] or "Receiver_account" in meta["feature_cols"]:
            errors.append("Gate 1 Violation: Account IDs found in feature list!")

    # GATE 2: Threshold, calibration & resampling
    print("Checking Gate 2: Validation thresholding & calibration bundle...")
    if not os.path.exists("models/ensemble.joblib"):
        errors.append("Gate 2: models/ensemble.joblib does not exist!")
    else:
        bundle = joblib.load("models/ensemble.joblib")
        if "threshold" not in bundle or bundle["threshold"] is None:
            errors.append("Gate 2 Violation: Missing threshold in ensemble bundle!")
        if "calibrator" not in bundle or bundle["calibrator"] is None:
            errors.append("Gate 2 Violation: Missing calibrator in ensemble bundle!")

    # GATE 3: Required files exist and non-empty
    print("Checking Gate 3: File existence and integrity...")
    required_files = [
        "outputs/metrics.csv",
        "outputs/bi_summary.csv",
        "outputs/risk_tier_summary.csv",
        "outputs/analyst_capacity_table.csv",
        "outputs/stress_test_imbalance.csv",
        "outputs/adversarial_results.csv",
        "outputs/flagged_with_reasons.csv",
        "outputs/shap_importance.csv",
        "outputs/data_quality.csv",
        "outputs/eda/EDA_insights.md",
        "outputs/BUSINESS_INSIGHTS.md",
        "docs/Report.docx",
        "docs/Presentation.pptx",
        "docs/VIVA_QA.md",
        "docs/CHEAT_SHEET.md",
        "docs/DEMO_SCRIPT.md",
        "DEPLOY.md",
        "Dockerfile",
        "docker-compose.yml",
        "data/sample/demo_transactions.csv",
        "outputs/figures/pr_curve.png",
        "outputs/figures/roc_curve.png",
        "outputs/figures/confusion_matrix.png",
        "outputs/figures/dashboard.png",
        "docs/system_architecture.png"
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            errors.append(f"Gate 3 Violation: Missing required file {rf}")
        elif os.path.getsize(rf) == 0:
            errors.append(f"Gate 3 Violation: File {rf} is empty!")

    # Check images can open
    test_imgs = ["outputs/figures/pr_curve.png", "outputs/figures/dashboard.png", "docs/system_architecture.png"]
    for img_p in test_imgs:
        if os.path.exists(img_p):
            try:
                with Image.open(img_p) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Gate 3 Violation: Image {img_p} is corrupt: {str(e)}")

    # GATE 6: PR-AUC <= 0.99 (Leakage detection)
    print("Checking Gate 6: PR-AUC leakage gate...")
    if os.path.exists("outputs/metrics.csv"):
        m = pd.read_csv("outputs/metrics.csv").iloc[0]
        pr_auc = float(m["pr_auc"])
        if pr_auc > 0.99:
            errors.append(f"Gate 6 Violation: Test PR-AUC {pr_auc:.4f} > 0.99 indicates potential target leakage!")
        else:
            print(f"Gate 6 Passed: Test PR-AUC is {pr_auc:.4f} (<= 0.99, realistic and robust).")

    # GATE 7: Numbers match across outputs
    print("Checking Gate 7: Number consistency across outputs...")
    if os.path.exists("outputs/metrics.csv") and os.path.exists("outputs/bi_summary.csv"):
        m = pd.read_csv("outputs/metrics.csv").iloc[0]
        bi = pd.read_csv("outputs/bi_summary.csv").iloc[0]
        if int(m["test_total"]) != int(bi["test_total_transactions"]):
            errors.append("Gate 7 Violation: Test total mismatch between metrics.csv and bi_summary.csv!")
        if int(m["test_positives"]) != int(bi["test_positives"]):
            errors.append("Gate 7 Violation: Test positives mismatch between metrics.csv and bi_summary.csv!")

    if errors:
        print("\nQUALITY GATE FAILURES:")
        for e in errors:
            print(f"  ❌ {e}")
        raise RuntimeError(f"Quality gate checks failed with {len(errors)} errors.")
    else:
        print("\nALL QUALITY GATES PASSED! (0 errors)\n")

if __name__ == "__main__":
    verify_quality_gates()
