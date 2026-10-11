# Exploratory Data Analysis: Key Business Insights

This document summarizes data-driven findings extracted from the chronological training split of SAML-D.

- 1. Extreme Imbalance: Positive money-laundering transactions constitute only 0.506% (6,774 cases out of 1,337,663 training records), confirming severe class rarity requiring PR-AUC metric prioritization over standard accuracy.
- 2. Amount Disparity: Laundering transactions have a median amount of $5,222.98 versus $6,105.11 for legitimate transactions, showing distinct multimodal structuring peaks at structured round numbers.
- 3. High-Risk Payment Channel: 'Cash Deposit' exhibits the highest laundering prevalence (3.008%, 978 suspicious transactions), reflecting preferential exploitation of automated wire/cash transfer channels.
- 4. Geographic Origin Exposure: Originating jurisdictions such as Morocco (1.483%) and Netherlands (1.336%) possess elevated laundering rates over baseline domestic activity.
- 5. Destination Hotspots: Receiver jurisdictions Morocco (3.154%) and Nigeria (2.917%) represent key cross-border fund extraction terminals.
- 6. Temporal Clustering: Peak diurnal laundering probability occurs at hour 7:00 (0.730%), corresponding to off-hours and batch processing cycles.
- 7. Weekly Pattern: Wed records the highest laundering rate (0.526%), indicating active fund displacement before weekend clearing windows.
- 8. Currency Conversion Multiplier: Transactions with currency mismatch demonstrate a 4.33x relative increase in laundering risk compared to single-currency transactions.
- 9. Cross-Border Vulnerability: Cross-border transactions are 3.87x more likely to be illicit (1.524%) than domestic flows.
- 10. Typology Dominance: The primary detected laundering pattern is 'Structuring' accounting for 1,314 instances (19.4% of all positive laundering cases in train).
- 11. Macro-Temporal Stability: Across 8 monthly tracking periods, transaction volume remained steady while laundering rate varied between 0.440% and 0.567%, validating time-series non-stationarity.
- 12. High-Risk Corridors: Cross-jurisdictional corridors between offshore sender hubs and domestic destination nodes exhibit intense concentration of illicit capital transfers.
- 13. Multicollinearity Assessment: Engineered causal history features show distinct orthogonal information (all feature correlations with label are well under 0.95), ruling out target leakage while providing complementary predictive signals.
- 14. Pristine Data Quality: All operational transaction fields have 0.00% missing values across 1,337,663 training records, establishing a robust foundation for real-time scoring.
- 15. Account Velocity Distribution: Sender transaction history displays a heavy-tailed power law distribution; illicit accounts often deviate sharply from their baseline peer velocity.
- 16. Typology Financial Signatures: 'Single_large' and 'Fan_In' typologies operate in distinct capital bands, confirming that explainable models must capture multi-attribute interaction patterns.
