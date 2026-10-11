import os
import sys
import argparse
import time
import importlib

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_pipeline(sample_ratio=None, force=False):
    t_start = time.time()
    print("=" * 70)
    print("EXPLAINABLE AML FRAUD DETECTION & RISK MANAGEMENT PIPELINE")
    print("Course: 22MDC56 Machine Learning Laboratory")
    print("=" * 70)

    # Stage 1: Preprocessing
    print("\n[Stage 1/7] Executing Preprocessing & Zero-Leakage Split...")
    stage1 = importlib.import_module("src.01_preprocess")
    if force and sample_ratio is None:
        for f in ["data/train.parquet", "data/val.parquet", "data/test.parquet", "models/preprocessing_meta.joblib"]:
            if os.path.exists(f):
                os.remove(f)
    stage1.run_preprocessing(sample_ratio=sample_ratio)

    # Stage 2: EDA
    print("\n[Stage 2/7] Generating Exploratory Data Analysis Charts & Insights...")
    stage2 = importlib.import_module("src.02_eda")
    stage2.run_eda()

    # Stage 3: Models
    print("\n[Stage 3/7] Training Baseline, LightGBM, Random Forest, & Ensemble...")
    stage3 = importlib.import_module("src.03_models")
    stage3.run_model_training()

    # Stage 4: Evaluation & Stress Tests
    print("\n[Stage 4/7] Running Test Evaluation, SHAP Transparency, & Stress Tests...")
    stage4 = importlib.import_module("src.04_evaluate")
    stage4.run_evaluation()

    # Stage 5: BI Decision Layer
    print("\n[Stage 5/7] Computing Expected Loss, Risk Tiers, & BI Optimization...")
    stage5 = importlib.import_module("src.05_bi_layer")
    stage5.run_bi_layer()

    # Stage 6: Architecture Diagram & Report
    print("\n[Stage 6/7] Building Architecture Diagrams, Presentation, & Academic Report...")
    from src.generate_architecture_diagram import generate_architecture_diagram
    generate_architecture_diagram()

    from src.generate_presentation import create_presentation
    create_presentation()

    from src.generate_report import create_report
    create_report()

    # Stage 7: Quality Verification Gate
    print("\n[Stage 7/7] Running Quality Gates & Automated Verification...")
    stage7 = importlib.import_module("src.99_verify")
    stage7.verify_quality_gates()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AML ML Pipeline end-to-end")
    parser.add_argument("--sample", action="store_true", help="Run fast pipeline on 5%% sample")
    parser.add_argument("--force", action="store_true", help="Force recomputing preprocessing from scratch")
    args = parser.parse_args()
    ratio = 0.05 if args.sample else None
    run_pipeline(sample_ratio=ratio, force=args.force)
