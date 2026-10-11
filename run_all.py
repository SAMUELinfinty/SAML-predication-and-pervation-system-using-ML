import os
import sys
import argparse
import time

def run_pipeline(sample_ratio=None):
    t_start = time.time()
    print("=" * 70)
    print("EXPLAINABLE AML FRAUD DETECTION & RISK MANAGEMENT PIPELINE")
    print("Course: 22MDC56 Machine Learning Laboratory")
    print("=" * 70)

    # Stage 1: Preprocessing
    print("\n[Stage 1/7] Executing Preprocessing & Zero-Leakage Split...")
    from src.01_preprocess import run_preprocessing
    run_preprocessing(sample_ratio=sample_ratio)

    # Stage 2: EDA
    print("\n[Stage 2/7] Generating Exploratory Data Analysis Charts & Insights...")
    from src.02_eda import run_eda
    run_eda()

    # Stage 3: Models
    print("\n[Stage 3/7] Training Baseline, LightGBM, Random Forest, & Ensemble...")
    from src.03_models import run_model_training
    run_model_training()

    # Stage 4: Evaluation & Stress Tests
    print("\n[Stage 4/7] Running Test Evaluation, SHAP Transparency, & Stress Tests...")
    from src.04_evaluate import run_evaluation
    run_evaluation()

    # Stage 5: BI Decision Layer
    print("\n[Stage 5/7] Computing Expected Loss, Risk Tiers, & BI Optimization...")
    from src.05_bi_layer import run_bi_layer
    run_bi_layer()

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
    from src.99_verify import verify_quality_gates
    verify_quality_gates()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run fast pipeline on 5% sample")
    args = parser.parse_args()
    ratio = 0.05 if args.sample else None
    run_pipeline(sample_ratio=ratio)
