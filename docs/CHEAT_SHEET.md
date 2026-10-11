# One-Page Executive Cheat Sheet: AML ML Pipeline

**Project:** Explainable ML Pipeline for AML Fraud Detection & Business Risk Management  
**Course:** 22MDC56 ML Laboratory | **Team:** Sameul D (2403717673721040), Mounica K (2403717673722028), Naveen raj (2403717673721033)

---

### Core Project Numbers at a Glance (Untouched Test Set Evaluation)
- **Scale:** 9,504,852 raw transactions; **Train:** 1,337,663 rows; **Val:** 1,430,805 rows; **Test:** 1,412,824 rows.
- **Prevalence:** 0.104% ground-truth positives (1,683 true positives in untouched test split).
- **M1 LightGBM Test PR-AUC:** **0.9032** (Best single model; optimized via scale_pos_weight).
- **M4 Ensemble Test PR-AUC:** **0.6292** | **ROC-AUC:** **0.9948** (Rank-normalized combination).
- **Recall@1.0% Review Budget:** **93.58%** of all laundering cases intercepted in top 1.0% workload budget.
- **Financial Fraud at Risk:** **$45,894,836.39**
- **Net Cost Savings:** **$44,951,157.66** (**97.94%** reduction vs unmitigated baseline).
- **Adversarial Robustness:** Clean PR-AUC maintained at **0.9824** after defense retraining against black-box evasion.
- **Temporal Stability:** First-half test PR-AUC (**0.6305**) vs Second-half test PR-AUC (**0.6291**).

---

### One-Line Justification for Every Architectural Design Choice

| Design Choice | One-Line Technical Justification |
| :--- | :--- |
| **Purging `Laundering_type`** | Prevents 100% artificial target leakage by removing post-hoc criminal typology labels. |
| **Chronological 70/15/15 Split** | Eliminates look-ahead bias inherent to randomized k-fold splits ($max(train) < min(val) < min(test)$). |
| **Shift-by-1 Causal History** | Guarantees customer velocity features use only transactions occurring strictly before timestamp $t$. |
| **Account IDs Excluded** | Prevents high-cardinality memorization and forces the model to learn transferable behavioral patterns. |
| **PR-AUC as Headline Metric** | Directly penalizes false-alarm alert fatigue in extreme class imbalance where ROC-AUC is misleading. |
| **Polars Data Engine** | Delivers 20x multithreaded speedup over pandas, processing 9.5M windowed calculations in seconds. |
| **Negative Downsampling (Train Only)** | Accelerates training efficiency while preserving all 6,774 positives; test remains 100% untouched. |
| **Rank Normalization Ensemble** | Unifies differing model score scales (LightGBM probabilities vs Isolation Forest depths) into percentiles. |
| **Isolation Forest Included** | Provides unsupervised protection against novel, zero-day laundering typologies absent from training labels. |
| **Isotonic Regression** | Maps non-linear score clusters into empirical posterior probabilities without assuming a rigid sigmoid. |
| **Validation Threshold Tuning** | Prevents test over-optimism by selecting both F1 and cost-optimal operating thresholds strictly on validation. |
| **Asymmetric Cost Matrix** | Models real banking economics: False Negative cost = Amount, False Positive review cost = $50. |
| **Expected Loss ($P \times \text{Amt}$)** | Prioritizes high-exposure financial risk over low-value alerts regardless of raw probability. |
| **SHAP TreeExplainer** | Satisfies regulatory explainability mandates (FATF/EU AMLD) with polynomial-time exact attribution. |
| **Top-3 Plain-English Reasons** | Translates high-dimensional math into actionable compliance justifications for automated SAR filings. |
| **Adversarial Retraining** | Neutralizes black-box evasion attacks by training on structured and amount-perturbed synthetic negatives. |
| **FastAPI + Streamlit Split** | Decouples sub-second scoring microservices from interactive business intelligence visualization. |
