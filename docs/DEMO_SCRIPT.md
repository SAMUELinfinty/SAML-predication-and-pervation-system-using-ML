# 3-4 Minute Evaluation & Demo Walkthrough Script

**Project:** Explainable ML Pipeline for AML Fraud Detection & Business Risk Management  
**Course:** 22MDC56 Machine Learning Laboratory  
**Team:** Sameul D (2403717673721040), Mounica K (2403717673722028), Naveen raj (2403717673721033)

---

## Demo Overview (3-4 Minutes Total)

This walkthrough demonstrates the end-to-end capabilities of our explainable AML pipeline across six distinct functional modules:
1. System Health & Executive Overview (30 sec)
2. Real-Time Single Transaction Scoring (90 sec)
3. High-Throughput Batch Portfolio Screening (30 sec)
4. Business Risk & Analyst Capacity Simulator (30 sec)
5. SHAP Transparency & Model Diagnostics (30 sec)
6. Adversarial Robustness & Stress Tests (30 sec)

---

### Step 1: Launch the Application (0:00 - 0:30)
1. Start the FastAPI backend and Streamlit frontend:
   ```bash
   python -m uvicorn app.backend.main:app --port 8000
   python -m streamlit run app/frontend/app.py --server.port 7860
   ```
2. Navigate to `http://localhost:7860`.
3. Check the left sidebar to confirm that **Backend: Connected (http://localhost:8000)** is glowing green.
4. Point out the top 4 headline KPI cards on the **Overview** page:
   - **M1 LightGBM Test PR-AUC:** `0.9032`
   - **Top 1.0% Recall@K:** `93.58%`
   - **Total Net Financial Savings:** `$44.95M`
   - **Operational Cost Reduction:** `97.94%`

---

### Step 2: Test Case A — Obvious High-Risk Laundering Alert (0:30 - 1:30)
1. In the sidebar, select **Score Single Transaction**.
2. Click the **🚨 Load High-Risk Sample** button (or manually enter the following values):
   - **Amount:** `31973.22`
   - **Payment Type:** `Debit card`
   - **Payment Currency:** `UK pounds`
   - **Received Currency:** `UK pounds`
   - **Sender Bank Location:** `UK`
   - **Receiver Bank Location:** `UK`
   - **Hour:** `7`
   - **Day of Week:** `4`
   - **Sender Prior Txn Count:** `5`
   - **Sender Prior Mean Amount:** `170162.18`
   - **Sender Prior Amount Std:** `338656.62`
   - **Receiver Prior In-Count:** `14`
   - **Sender Distinct Prior Receivers:** `4`
3. Click **⚡ Evaluate Transaction Risk**.
4. **Verified Live Outputs:**
   - **Assigned Risk Tier:** `High` / `Critical` (badge in red/orange)
   - **Calibrated Laundering Probability:** `> 39%`
   - **Expected Financial Loss:** `>$3,000`
   - **Recommended Action:** `Flag for Priority Review` / `Block & Escalate`
   - **Top-3 Plain-English Reasons:**
     1. *High-risk execution channel (Cash/Wire/Cross-border)*
     2. *Fan-out dispersion across multiple distinct counterparty accounts*
     3. *Unusually high amount deviation scale vs historical velocity*

---

### Step 3: Test Case B — Obvious Low-Risk Legitimate Transaction (1:30 - 2:15)
1. On the same page, click **✅ Load Low-Risk Sample** (or enter):
   - **Amount:** `45.50`
   - **Payment Type:** `Debit card`
   - **Payment Currency:** `UK pounds`
   - **Received Currency:** `UK pounds`
   - **Sender Bank Location:** `UK`
   - **Receiver Bank Location:** `UK`
   - **Hour:** `14` (2:00 PM standard banking hours)
   - **Day of Week:** `2` (Tuesday)
   - **Sender Prior Count:** `50`
   - **Sender Prior Mean:** `55.00`
   - **Sender Prior Std:** `20.00`
2. Click **⚡ Evaluate Transaction Risk**.
3. **Verified Live Outputs:**
   - **Assigned Risk Tier:** `Low` (badge in green)
   - **Calibrated Probability:** `< 0.05%`
   - **Expected Loss:** `< $1.00`
   - **Recommended Action:** `Automated Approval`

---

### Step 4: Batch Portfolio Screening (2:15 - 2:45)
1. Navigate to **Batch Scoring & Screening** in the sidebar.
2. Click **📂 Load Demo Sample File (5,000 records)**.
3. Click **🚀 Execute Batch Screening**.
4. Point out the sub-second processing and summary metrics:
   - High-risk cases segregated into Critical/High tiers.
   - Live download button enabled: **📥 Download Scored Transactions CSV**.

---

### Step 5: BI & Cost Simulator (2:45 - 3:15)
1. Navigate to **BI & Cost Simulator**.
2. Drag the **Analyst Review Budget** slider to `1.0%`.
3. Set **Cost Per Investigation** to `$50.00`.
4. Point out the live API simulation:
   - **Laundering Cases Caught:** `93.58%`
   - **Intercepted Capital:** `$44.7M`
   - **Net Financial Savings:** `$43.99M` after deducting investigator review salaries!

---

### Step 6: Explainability & Robustness Wrap-Up (3:15 - 3:45)
1. In **EDA & Results Gallery**, showcase the 16 EDA plots and SHAP Beeswarm.
2. In **Robustness & Stress Tests**, highlight that our model maintains a `0.9824` clean PR-AUC after adversarial retraining against black-box amount-scaling evasion attempts.
