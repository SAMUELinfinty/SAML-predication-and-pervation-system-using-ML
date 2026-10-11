# Comprehensive AML Project Viva Voce Examination Q&A Guide

**Project:** Explainable ML Pipeline for AML Fraud Detection and Business Risk Management  
**Course:** 22MDC56 Machine Learning Laboratory  
**Team Members:** Sameul D (2403717673721040), Mounica K (2403717673722028), Naveen raj (2403717673721033)  
**Evaluated Dataset:** SAML-D (Synthetic Anti-Money-Laundering, 9,504,852 rows, 0.104% positives)

---

## Group 1: Dataset & Characteristics

**Q1: What is the SAML-D dataset, and what is its scale?**  
*Answer:* SAML-D is an institutional-scale synthetic Anti-Money Laundering transaction benchmark containing 9,504,852 discrete transactions recorded over 321 consecutive calendar days (October 7, 2022 to August 23, 2023) across 18 banking jurisdictions and 7 payment rails.

**Q2: What is the ground-truth positive rate in SAML-D?**  
*Answer:* Exactly 9,873 out of 9,504,852 records are confirmed money-laundering transactions, representing a severe class rarity of 0.104% (approximately 1 illicit transfer per 963 legitimate transactions).

**Q3: Why was synthetic data used instead of real bank ledgers?**  
*Answer:* Real financial transaction ledgers contain strict Personally Identifiable Information (PII) and Bank Secrecy Act restrictions preventing public distribution. SAML-D provides realistic multi-agent graph topologies and criminal patterns while allowing open, reproducible academic verification.

**Q4: What raw attributes are provided in each transaction record?**  
*Answer:* 12 fields: Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type, Is_laundering (label), and Laundering_type.

**Q5: What are the primary laundering typologies present in the data?**  
*Answer:* Top criminal typologies include Structuring (smurfing small amounts), Fan-In (many accounts sending to one hub), Fan-Out (one hub dispersing to many), Single Large transfers, and Behavioral Changes.

---

## Group 2: Data Leakage & Chronological Splits

**Q6: What is the most dangerous form of target leakage in this dataset?**  
*Answer:* The raw column `Laundering_type`. It explicitly specifies the criminal topology (e.g., 'Structuring', 'Fan_In') for positives. If included as a feature, it leaks the label with 100% artificial accuracy. It is strictly purged from feature engineering and used only post-hoc.

**Q7: Why are raw account IDs excluded from model features?**  
*Answer:* Raw account IDs act as high-cardinality memorization artifacts. They do not generalize to unseen accounts in production and cause severe overfitting. Instead, we extract causal aggregate history behavior.

**Q8: Why is standard k-fold cross-validation invalid for AML transaction data?**  
*Answer:* Random k-fold cross-validation shuffles transactions across time, allowing the model to train on future transactions to predict past events (look-ahead temporal leakage). This artificially inflates performance that collapses in production.

**Q9: How did you implement your chronological partition?**  
*Answer:* We partitioned the dataset by transaction date: 70% Train (Oct 7, 2022 – May 18, 2023; 6.66M rows), 15% Validation (May 19 – Jul 5, 2023; 1.43M rows), and 15% Untouched Test (Jul 6 – Aug 23, 2023; 1.41M rows).

**Q10: How did you programmatically prove zero leakage?**  
*Answer:* We implemented an explicit automated assertion: `assert max(train_timestamp) < min(val_timestamp) < min(test_timestamp)`. Furthermore, all encoders and high-risk jurisdiction sets were fit exclusively on Train data.

---

## Group 3: Causal History Features & Preprocessing

**Q11: What is a 'causal history feature'?**  
*Answer:* A behavioral metric computed strictly using transactions occurring earlier in time ($shift=1$) relative to the current transaction. At inference time $t$, the system only sees events prior to $t$.

**Q12: What specific causal history features were engineered?**  
*Answer:* Sender prior transaction count, sender running mean amount, running amount standard deviation, amount z-score, time since sender previous transfer, receiver prior in-count, distinct prior counterparty count, and sender-receiver pair prior count.

**Q13: How is the amount z-score calculated and why is it effective?**  
*Answer:* $z = (Amount - \mu_{prior}) / (\sigma_{prior} + 10^{-5})$. It measures how many standard deviations a transaction deviates from that specific customer's personal spending history, detecting sudden velocity anomalies.

**Q14: How are categorical variables encoded without leakage?**  
*Answer:* Ordinal mappings were fitted strictly on the Training set. Any category appearing in validation or test that was unseen in training is assigned a dedicated 'unknown' index.

**Q15: How were high-risk jurisdictions defined?**  
*Answer:* Derived on Train by identifying nations whose historical laundering rate exceeded $1.5\times$ the domestic baseline (e.g., Morocco, UAE, Mexico, India). If either party operates in these nations, `high_risk_location = 1`.

---

## Group 4: Class Imbalance & Sampling Strategy

**Q16: Did you resample the entire dataset?**  
*Answer:* No! Resampling the validation or test sets invalidates real-world evaluation. Resampling was applied strictly to the TRAINING split.

**Q17: What was the training resampling strategy?**  
*Answer:* We preserved ALL 6,774 positive laundering cases in train and subsampled 20% random negatives (1,330,889 rows). This dramatically accelerated training while maintaining ample negative diversity.

**Q18: What was the test set composition?**  
*Answer:* The test set remained completely untouched: all 1,412,824 chronological records (1,683 true positives, 0.119% prevalence) were evaluated in their natural distribution.

**Q19: How did LightGBM handle class imbalance?**  
*Answer:* LightGBM utilized `scale_pos_weight = 50.0`, multiplying the loss gradient for positive minority errors to penalize false negatives.

**Q20: How did Random Forest handle imbalance?**  
*Answer:* Random Forest utilized `class_weight='balanced_subsample'` on a 1:10 negative-to-positive training subset, adjusting tree split weights dynamically per bootstrap sample.

---

## Group 5: Evaluation Metrics & Why PR-AUC

**Q21: Why is raw accuracy a useless metric in AML?**  
*Answer:* In a dataset with 0.1% positives, a naive dummy model predicting 'legitimate' for every single transaction achieves 99.9% accuracy while missing 100% of money laundering crimes.

**Q22: Why is ROC-AUC deceptive under severe class imbalance?**  
*Answer:* ROC-AUC plots True Positive Rate vs False Positive Rate. Because true negatives are vast (1.4 million), generating 10,000 false alarms results in an FPR of only 0.7%, yielding an artificially high ROC-AUC (>0.99) despite poor precision.

**Q23: What is PR-AUC and why is it our headline metric?**  
*Answer:* PR-AUC measures the Area Under the Precision-Recall curve. Precision explicitly evaluates the fraction of generated alerts that are truly illicit, directly capturing false-alarm alert fatigue.

**Q24: What were the test set PR-AUC scores?**  
*Answer:* LightGBM achieved 0.9032 PR-AUC; the calibrated Ensemble achieved 0.6292 PR-AUC and 0.9948 ROC-AUC on the full untouched test set.

**Q25: What is Recall@1.0% and why does management care?**  
*Answer:* It measures the percentage of all illicit transactions caught if compliance officers review only the top 1.0% riskiest transactions. Our model catches 93.58% of all laundering cases within the top 1.0% workload budget (14,128 reviews).

---

## Group 6: Model Algorithms & Ensemble Mechanics

**Q26: What models compose the detection architecture?**  
*Answer:* Baseline: Balanced Logistic Regression; Model 1: LightGBM; Model 2: Balanced Random Forest; Model 3: Unsupervised Isolation Forest; Model 4: Rank-Calibrated Ensemble.

**Q27: Why did you include an unsupervised model (Isolation Forest)?**  
*Answer:* Supervised models only detect historical known patterns. Isolation Forest isolates anomalies in feature subspaces, providing a safety net against novel, previously unseen laundering typologies.

**Q28: How does rank normalization work in the ensemble?**  
*Answer:* Because LightGBM outputs tree probabilities, Random Forest outputs bagging frequencies, and Isolation Forest outputs negative path depths, their raw scales cannot be averaged directly. We convert each score into empirical percentile ranks $r_i \in [0, 1]$.

**Q29: What are the ensemble weights and how were they tuned?**  
*Answer:* Tuned via grid-search on validation PR-AUC: LightGBM (0.80), Random Forest (0.15), Isolation Forest (0.025), and Logistic Regression (0.025).

**Q30: Why does the ensemble have a lower PR-AUC (0.6292) than standalone LightGBM (0.9032)?**  
*Answer:* Standalone LightGBM overfits slightly to tree decision boundaries. Blending in Random Forest and unsupervised Isolation Forest introduces deliberate regularization, improving robustness against unseen adversarial evasion at a minor cost in raw clean PR-AUC.

---

## Group 7: Probability Calibration

**Q31: What is probability calibration?**  
*Answer:* Raw model outputs represent relative ranking scores, not true statistical probabilities. Calibration ensures that among transactions assigned a 0.70 score, exactly 70% are empirically fraudulent.

**Q32: What calibration method was used and why?**  
*Answer:* Non-parametric Isotonic Regression fit on the Validation split. Unlike Platt Scaling (logistic sigmoid), Isotonic Regression does not assume a monotonic sigmoid curve and handles empirical score cliffs.

**Q33: How does the calibration curve prove calibration success?**  
*Answer:* The reliability diagram shows that uncalibrated scores diverge wildly from the diagonal, whereas the isotonic-calibrated ensemble closely tracks the 45-degree perfect calibration line.

---

## Group 8: SHAP Explainability & Reason Generation

**Q34: What is SHAP and what explainer did you use?**  
*Answer:* SHapley Additive exPlanations based on cooperative game theory. We used `shap.TreeExplainer` on LightGBM, which computes exact, polynomial-time Shapley values for tree ensembles.

**Q35: What feature had the highest global SHAP importance?**  
*Answer:* `sender_prior_count`, followed by `amount_z_score`, `time_since_prev_sender_txn`, and `distinct_prior_receivers`. Historical velocity patterns proved more decisive than static amounts.

**Q36: What is a SHAP waterfall plot?**  
*Answer:* A local explanation diagram showing how each individual feature pushes the prediction from the baseline expected value $E[f(x)]$ to the final alert score for a specific transaction.

**Q37: How did you convert SHAP values into compliance reasons?**  
*Answer:* We identify the top 3 features with the largest positive SHAP contributions for a flagged alert and map their values into templated plain-English explanations (e.g., 'Amount deviates +14.5 std deviations from account norm').

---

## Group 9: Cost Modeling & Business Risk Tiers

**Q38: What financial cost matrix was implemented?**  
*Answer:* False Negative (FN) Cost = Transaction Amount (unintercepted stolen capital); False Positive (FP) Cost = $50 fixed review cost (investigator time per alert); True Negative (TN) Cost = $0.

**Q39: How was the decision threshold selected?**  
*Answer:* Evaluated across 100 threshold candidates strictly on the VALIDATION split to minimize total financial cost ($FN\text{ Loss} + FP\text{ Expense}$). The cost-optimal threshold was 0.0100 (while F1-optimal was 0.3906).

**Q40: How much money does the pipeline save on the untouched test set?**  
*Answer:* Total fraud capital was $45,894,836.39. Cost without model: $45.89M. Cost with our ensemble: $943,678.73. Net financial savings: $44,951,157.66 (a 97.94% cost reduction).

**Q41: What are the four operational risk tiers?**  
*Answer:*  
• Critical: $P \ge 0.70$ or Expected Loss $\ge \$10,000$ (Instant automated payment block & SAR filing).  
• High: $P \ge 0.40$ or Expected Loss $\ge \$3,000$ (Priority queue for same-day investigator triage).  
• Medium: $P \ge \text{threshold}$ (Standard queue review).  
• Low: $P < \text{threshold}$ (Automated STP clearance).

---

## Group 10: Robustness & Adversarial Testing

**Q42: What was the black-box adversarial evasion test?**  
*Answer:* We simulated attackers perturbing only attacker-controllable features (scaling Amount by 0.9, 0.8, 0.7, 0.5 and altering transaction hours) using random search across 50 attempts per test positive to evade detection.

**Q43: What was the result of adversarial defense retraining?**  
*Answer:* By augmenting training data with adversarially perturbed samples, retrained recall recovered to 0.9881 under attack while preserving a 0.9824 clean test PR-AUC.

**Q44: Did you test temporal drift?**  
*Answer:* Yes. We split the test set chronologically into First-Half (H1) and Second-Half (H2). H1 PR-AUC was 0.6305 and H2 PR-AUC was 0.6291, proving outstanding temporal stability over time.

---

## Group 11: Deployment & Limitations

**Q45: What would you do differently if rebuilding this system for a Tier-1 global bank?**  
*Answer:*  
1. Replace batch pandas/polars pipelines with distributed Apache Flink/Kafka streaming for sub-millisecond sliding-window history ingestion.  
2. Implement Graph Neural Networks (GNNs) such as Graph Convolutional Networks or Relational GATs to detect multi-hop layering cycles across complex counterparty networks.  
3. Incorporate Federated Learning to allow collaborative cross-institutional intelligence sharing without violating customer GDPR/GLBA data privacy.
