import os
import io
import requests
import pandas as pd
import streamlit as st
from PIL import Image

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Explainable AML Fraud Detection & Risk BI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 18px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-card h2 {
        margin: 8px 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .tier-badge-critical {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-high {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-medium {
        background-color: #fef9c3;
        color: #a16207;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-low {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .reason-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check backend health
@st.cache_data(ttl=5)
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            return True, r.json()
        return False, None
    except Exception:
        return False, None

is_online, backend_meta = check_backend()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("AML Risk Navigator")
    st.caption("Course 22MDC56 ML Laboratory")
    
    # Status indicator
    if is_online:
        st.success(f"Backend: Connected ({BACKEND_URL})")
    else:
        st.error(f"Backend: Offline ({BACKEND_URL})")
        st.info("Start backend via: `python app/backend/main.py` or check Docker container.")

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "Overview & Dashboard",
            "Score Single Transaction",
            "Batch Scoring & Screening",
            "BI & Cost Simulator",
            "EDA & Results Gallery",
            "Robustness & Stress Tests"
        ]
    )
    st.markdown("---")
    st.markdown("**Team Members:**\n- Sameul D (2403717673721040)\n- Mounica K (2403717673722028)\n- Naveen raj (2403717673721033)")

# PAGE 1: OVERVIEW & DASHBOARD
if menu == "Overview & Dashboard":
    st.markdown("<div class='main-header'>Explainable ML Pipeline for AML Fraud Detection & Risk Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>End-to-End Synthetic Anti-Money-Laundering (SAML-D) Detection, SHAP Explainability, and Cost-Optimized Business Risk Decisions</div>", unsafe_allow_html=True)

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class='metric-card'>
            <h4>M1 LightGBM Test PR-AUC</h4>
            <h2>0.9032</h2>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card'>
            <h4>Top 1.0% Recall@K</h4>
            <h2>93.58%</h2>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card'>
            <h4>Total Financial Savings</h4>
            <h2>$44.95M</h2>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='metric-card'>
            <h4>Cost Reduction vs Baseline</h4>
            <h2>97.94%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Core Architecture & Highlights
    col_arch, col_highlights = st.columns([1.2, 1])
    with col_arch:
        st.subheader("System Architecture & Workflow")
        if os.path.exists("outputs/figures/dashboard.png"):
            st.image("outputs/figures/dashboard.png", caption="Executive 2x2 Risk Decision & Financial Optimization Dashboard", use_container_width=True)
        else:
            st.info("Run `python src/05_bi_layer.py` to generate the dashboard image.")

    with col_highlights:
        st.subheader("Pipeline Innovations (Addressing Research Gaps)")
        st.markdown("""
        - **Zero Temporal Leakage:** Strict chronological split ($max(train) < min(val) < min(test)$), causal account history engineered with strictly earlier transactions only ($shift=1$).
        - **Multi-Model Rank-Calibrated Ensemble:** Weighted rank normalization blending LightGBM, Random Forest, Isolation Forest, and Logistic Regression with validation Isotonic Regression.
        - **Post-Hoc SHAP Transparency:** TreeExplainer provides local credit/debit attribution and translates complex feature spaces into top-3 plain English explanations for audit compliance.
        - **Cost-Optimal BI Decision Layer:** Connects raw ML predictions to operational constraints: False Negative cost = Amount, False Positive review cost = $50. Optimizes triage thresholds to save $44.95M.
        - **Adversarial & Imbalance Robustness:** Validated across severe class imbalances (1:1 to natural 1000:1) and black-box evasion attacks with adversarial defense recovery.
        """)

# PAGE 2: SCORE A SINGLE TRANSACTION
elif menu == "Score Single Transaction":
    st.markdown("<div class='main-header'>Real-Time Transaction Scoring & Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Submit transaction attributes for sub-second risk classification, tier allocation, expected loss, and plain-English SHAP attribution.</div>", unsafe_allow_html=True)

    if not is_online:
        st.warning("⚠️ FastAPI Backend is offline. Please start it at `http://localhost:8000` to enable scoring.")

    # Preset sample buttons
    b_col1, b_col2, b_col3 = st.columns([1, 1, 3])
    preset_data = {}
    with b_col1:
        if st.button("🚨 Load High-Risk Sample"):
            preset_data = {
                "Amount": 31973.22,
                "Payment_type": "Debit card",
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "hour": 7,
                "day_of_week": 4,
                "is_weekend": 0,
                "sender_prior_count": 5,
                "sender_prior_mean": 170162.18,
                "sender_prior_std": 338656.62,
                "amount_z_score": -0.408,
                "time_since_prev_sender_txn": 167.1,
                "receiver_prior_in_count": 14,
                "sender_receiver_prior_count": 0,
                "distinct_prior_receivers": 4
            }
    with b_col2:
        if st.button("✅ Load Low-Risk Sample"):
            preset_data = {
                "Amount": 45.50,
                "Payment_type": "Debit card",
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "sender_prior_count": 50,
                "sender_prior_mean": 55.0,
                "sender_prior_std": 20.0,
                "amount_z_score": -0.4,
                "time_since_prev_sender_txn": 72.0,
                "receiver_prior_in_count": 120,
                "sender_receiver_prior_count": 15,
                "distinct_prior_receivers": 10
            }

    with st.form("single_tx_form"):
        st.subheader("Transaction Details")
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            amount = st.number_input("Amount ($ / Units)", min_value=1.0, max_value=5000000.0, value=preset_data.get("Amount", 5000.0), step=100.0)
            payment_type = st.selectbox("Payment Type", ["Cross-border", "Cash Deposit", "Cash Withdrawal", "Cheque", "ACH", "Debit card", "Credit card"], index=0 if "Payment_type" not in preset_data else ["Cross-border", "Cash Deposit", "Cash Withdrawal", "Cheque", "ACH", "Debit card", "Credit card"].index(preset_data["Payment_type"]))
        with f_c2:
            p_curr = st.selectbox("Payment Currency", ["UK pounds", "US dollar", "Euro", "Dirham", "Indian rupee", "Swiss franc", "Yen", "Moroccan dirham", "Turkish lira"], index=0 if "Payment_currency" not in preset_data else ["UK pounds", "US dollar", "Euro", "Dirham", "Indian rupee", "Swiss franc", "Yen", "Moroccan dirham", "Turkish lira"].index(preset_data["Payment_currency"]))
            r_curr = st.selectbox("Received Currency", ["UK pounds", "US dollar", "Euro", "Dirham", "Indian rupee", "Swiss franc", "Yen", "Moroccan dirham", "Turkish lira"], index=0 if "Received_currency" not in preset_data else ["UK pounds", "US dollar", "Euro", "Dirham", "Indian rupee", "Swiss franc", "Yen", "Moroccan dirham", "Turkish lira"].index(preset_data["Received_currency"]))
        with f_c3:
            s_loc = st.selectbox("Sender Bank Location", ["UK", "USA", "UAE", "Germany", "France", "Spain", "Italy", "Morocco", "India", "Albania", "Mexico", "Switzerland"], index=0 if "Sender_bank_location" not in preset_data else ["UK", "USA", "UAE", "Germany", "France", "Spain", "Italy", "Morocco", "India", "Albania", "Mexico", "Switzerland"].index(preset_data["Sender_bank_location"]))
            r_loc = st.selectbox("Receiver Bank Location", ["UK", "USA", "UAE", "Germany", "France", "Spain", "Italy", "Morocco", "India", "Albania", "Mexico", "Switzerland"], index=0 if "Receiver_bank_location" not in preset_data else ["UK", "USA", "UAE", "Germany", "France", "Spain", "Italy", "Morocco", "India", "Albania", "Mexico", "Switzerland"].index(preset_data["Receiver_bank_location"]))

        with st.expander("Timing & Causal History Parameters (Optional - Auto-filled by core banking gateway)"):
            h_c1, h_c2, h_c3 = st.columns(3)
            with h_c1:
                hour = st.slider("Hour of Day", 0, 23, preset_data.get("hour", 12))
                day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, preset_data.get("day_of_week", 3))
                is_weekend = st.selectbox("Is Weekend", [0, 1], index=preset_data.get("is_weekend", 0))
            with h_c2:
                s_cnt = st.number_input("Sender Prior Txn Count", min_value=0, value=preset_data.get("sender_prior_count", 15))
                s_mean = st.number_input("Sender Prior Mean Amount", min_value=0.0, value=preset_data.get("sender_prior_mean", 1200.0))
                s_std = st.number_input("Sender Prior Amount Std", min_value=0.0, value=preset_data.get("sender_prior_std", 350.0))
            with h_c3:
                r_in_cnt = st.number_input("Receiver Prior In-Count", min_value=0, value=preset_data.get("receiver_prior_in_count", 20))
                pair_cnt = st.number_input("Sender-Receiver Pair Prior Count", min_value=0, value=preset_data.get("sender_receiver_prior_count", 1))
                dist_rx = st.number_input("Sender Distinct Prior Receivers", min_value=0, value=preset_data.get("distinct_prior_receivers", 5))

        submit = st.form_submit_button("⚡ Evaluate Transaction Risk", type="primary")

    if submit and is_online:
        payload = {
            "Amount": float(amount),
            "Payment_type": payment_type,
            "Payment_currency": p_curr,
            "Received_currency": r_curr,
            "Sender_bank_location": s_loc,
            "Receiver_bank_location": r_loc,
            "hour": int(hour),
            "day_of_week": int(day_of_week),
            "is_weekend": int(is_weekend),
            "sender_prior_count": int(s_cnt),
            "sender_prior_mean": float(s_mean),
            "sender_prior_std": float(s_std),
            "receiver_prior_in_count": int(r_in_cnt),
            "sender_receiver_prior_count": int(pair_cnt),
            "distinct_prior_receivers": int(dist_rx)
        }
        try:
            with st.spinner("Analyzing multi-model ensemble and computing SHAP attribution..."):
                resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown("### Risk Evaluation Assessment")
                rc1, rc2, rc3, rc4 = st.columns(4)
                tier = result["risk_tier"]
                badge_class = f"tier-badge-{tier.lower()}"

                with rc1:
                    st.markdown(f"**Assigned Risk Tier:**<br><span class='{badge_class}'>{tier}</span>", unsafe_allow_html=True)
                with rc2:
                    st.metric("Calibrated Laundering Probability", f"{result['calibrated_prob']*100:.2f}%")
                with rc3:
                    st.metric("Expected Financial Loss", f"${result['expected_loss']:,.2f}")
                with rc4:
                    st.markdown(f"**Recommended Action:**<br><b>{result['decision']}</b>", unsafe_allow_html=True)

                st.progress(float(result["calibrated_prob"]))

                st.markdown("#### Top-3 Plain-English SHAP Attribution Reasons")
                for r in result["top_3_reasons"]:
                    st.markdown(f"<div class='reason-box'>🔍 <b>{r}</b></div>", unsafe_allow_html=True)
            else:
                st.error(f"Error from backend: {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with backend: {str(e)}")

# PAGE 3: BATCH SCORING
elif menu == "Batch Scoring & Screening":
    st.markdown("<div class='main-header'>High-Throughput Batch Transaction Screening</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload transaction portfolios (up to 50,000 records) to screen en masse, extract tier distributions, and export flagged files.</div>", unsafe_allow_html=True)

    c_up, c_demo = st.columns([2, 1])
    uploaded_file = None
    with c_up:
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    with c_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 Load Demo Sample File (5,000 records)"):
            demo_csv = "data/sample/demo_transactions.csv"
            if os.path.exists(demo_csv):
                with open(demo_csv, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
                    uploaded_file.name = "demo_transactions.csv"
                st.success("Loaded demo_transactions.csv successfully!")
            else:
                st.warning("Demo file not found.")

    if uploaded_file is not None and is_online:
        if st.button("🚀 Execute Batch Screening", type="primary"):
            with st.spinner("Processing batch with vectorized LightGBM & Isotonic Calibration..."):
                files = {"file": (getattr(uploaded_file, "name", "upload.csv"), uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read(), "text/csv")}
                resp = requests.post(f"{BACKEND_URL}/predict_batch", files=files, timeout=30)
            if resp.status_code == 200:
                b_res = resp.json()
                st.success(f"Successfully screened {b_res['total_scored']:,} transactions!")

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Critical Alerts", f"{b_res['critical_count']:,}")
                k2.metric("High Alerts", f"{b_res['high_count']:,}")
                k3.metric("Medium Alerts", f"{b_res['medium_count']:,}")
                k4.metric("Low (Approved)", f"{b_res['low_count']:,}")
                k5.metric("Total Expected Loss", f"${b_res['total_expected_loss_usd']:,.2f}")

                scored_df = pd.read_csv(io.StringIO(b_res["csv_data"]))
                st.dataframe(scored_df[["Amount", "Payment_type", "Sender_bank_location", "Receiver_bank_location", "calibrated_prob", "expected_loss", "risk_tier", "decision"]].head(100), use_container_width=True)

                st.download_button(
                    label="📥 Download Scored Transactions CSV",
                    data=b_res["csv_data"],
                    file_name="aml_screened_transactions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Batch screening error: {resp.text}")

# PAGE 4: BI & COST SIMULATOR
elif menu == "BI & Cost Simulator":
    st.markdown("<div class='main-header'>Business Risk & Analyst Capacity Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Dynamically balance investigator headcount, operational review budgets, and financial fraud mitigation yield.</div>", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.5])
    with sim_col1:
        st.subheader("Simulation Inputs")
        cap_slider = st.slider("Analyst Review Budget (% of Total Transactions)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        cost_slider = st.number_input("Cost Per Investigation Review ($ USD)", min_value=10.0, max_value=250.0, value=50.0, step=5.0)

        if is_online:
            sim_resp = requests.post(f"{BACKEND_URL}/bi_simulate", json={"capacity_pct": cap_slider, "review_cost_usd": cost_slider})
            if sim_resp.status_code == 200:
                s_data = sim_resp.json()
                st.markdown("### Projected Operational Impact")
                st.metric("Laundering Cases Caught (Recall)", f"{s_data['projected_recall']*100:.2f}%")
                st.metric("Total Intercepted Illicit Capital", f"${s_data['caught_illicit_amount_usd']:,.2f}")
                st.metric("Investigator Review Expenditures", f"${s_data['investigation_cost_usd']:,.2f}")
                st.metric("Net Financial Savings", f"${s_data['net_financial_savings_usd']:,.2f}")
                st.metric("Cost Reduction vs Zero-Model Baseline", f"{s_data['pct_cost_saved']:.2f}%")

    with sim_col2:
        st.subheader("Empirical Capacity Tradeoff Frontier")
        if os.path.exists("outputs/analyst_capacity_table.csv"):
            cap_table = pd.read_csv("outputs/analyst_capacity_table.csv")
            st.dataframe(cap_table, use_container_width=True)
        if os.path.exists("outputs/figures/dashboard.png"):
            st.image("outputs/figures/dashboard.png", caption="Cost Minimization & Illicit Capital Interception Curves", use_container_width=True)

# PAGE 5: EDA & RESULTS GALLERY
elif menu == "EDA & Results Gallery":
    st.markdown("<div class='main-header'>Exploratory Data Analysis & Model Evaluation Gallery</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Publication-grade visualizations generated strictly on chronological splits.</div>", unsafe_allow_html=True)

    cat = st.selectbox("Select Visual Category", ["EDA: Data Patterns & Laundering Rates", "Model Performance & Diagnostics", "SHAP Explainability Artifacts"])

    if cat == "EDA: Data Patterns & Laundering Rates":
        eda_files = sorted([f for f in os.listdir("outputs/eda") if f.endswith(".png")]) if os.path.exists("outputs/eda") else []
        if eda_files:
            selected_chart = st.selectbox("Select EDA Chart", eda_files)
            st.image(f"outputs/eda/{selected_chart}", use_container_width=True)
            if os.path.exists("outputs/eda/EDA_insights.md"):
                with st.expander("Read Corresponding Business Insights"):
                    with open("outputs/eda/EDA_insights.md", "r", encoding="utf-8") as f:
                        st.markdown(f.read())
        else:
            st.info("No EDA charts found. Run `python src/02_eda.py`.")

    elif cat == "Model Performance & Diagnostics":
        perf_charts = ["pr_curve.png", "roc_curve.png", "confusion_matrix.png", "model_comparison_bars.png", "calibration_curve.png"]
        sel_perf = st.selectbox("Select Diagnostic Plot", perf_charts)
        path = f"outputs/figures/{sel_perf}"
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info(f"{sel_perf} not found.")

        if os.path.exists("outputs/metrics.csv"):
            st.markdown("### Comprehensive Test Set Metrics")
            st.dataframe(pd.read_csv("outputs/metrics.csv"), use_container_width=True)

    elif cat == "SHAP Explainability Artifacts":
        shap_charts = ["shap_summary_bar.png", "shap_beeswarm.png", "shap_waterfall_tp.png", "shap_waterfall_fp.png"]
        sel_shap = st.selectbox("Select SHAP Figure", shap_charts)
        path = f"outputs/figures/{sel_shap}"
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info(f"{sel_shap} not found.")

        if os.path.exists("outputs/shap_importance.csv"):
            st.markdown("### Global Feature Impact (Mean |SHAP|)")
            st.dataframe(pd.read_csv("outputs/shap_importance.csv"), use_container_width=True)

# PAGE 6: ROBUSTNESS & STRESS TESTS
elif menu == "Robustness & Stress Tests":
    st.markdown("<div class='main-header'>Adversarial & Imbalance Robustness Stress Tests</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Systematic stress-testing under extreme class imbalance non-stationarity and malicious adversarial evasion attacks.</div>", unsafe_allow_html=True)

    r_c1, r_c2 = st.columns(2)
    with r_c1:
        st.subheader("Stress Test A: Severe Class Imbalance")
        st.markdown("Evaluation of model PR-AUC and ROC-AUC when positive prevalence varies across test distributions:")
        if os.path.exists("outputs/stress_test_imbalance.csv"):
            imb_df = pd.read_csv("outputs/stress_test_imbalance.csv")
            st.dataframe(imb_df, use_container_width=True)
            st.bar_chart(imb_df.set_index("ratio_name")[["pr_auc", "roc_auc"]])
        else:
            st.info("Imbalance stress test results not found.")

    with r_c2:
        st.subheader("Stress Test B: Black-Box Adversarial Evasion & Defense")
        st.markdown("Simulating attackers who maliciously alter amount structuring and execution timing to evade detection, followed by adversarial retraining defense:")
        if os.path.exists("outputs/adversarial_results.csv"):
            adv_df = pd.read_csv("outputs/adversarial_results.csv")
            st.dataframe(adv_df, use_container_width=True)
            st.markdown("""
            **Defense Takeaways:**
            - Attackers perturbing transaction amount and hours create evasion pressure on naive thresholds.
            - Adversarial data augmentation during model retraining restores illicit case detection while preserving clean generalization on unperturbed legitimate transactions.
            """)
        else:
            st.info("Adversarial results not found.")
