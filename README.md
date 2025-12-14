## Credit Scoring Business Understanding

**1. Influence of the Basel II Capital Accord on Model Interpretability**
The Basel II Capital Accord mandates that financial institutions maintain a rigorous framework for measuring and managing risk to ensure sufficient capital reserves. This regulatory requirement heavily influences our modeling approach by prioritizing **interpretability** and **auditability**. We cannot rely solely on "black box" models; we must be able to explain *why* a specific risk score was assigned to a customer to comply with regulatory standards for transparency. Consequently, our model selection and feature engineering (specifically Weight of Evidence) must produce transparent risk drivers that align with the Accord’s risk measurement guidelines.

**2. The Necessity and Risks of the Proxy Variable**
Since the provided dataset lacks a direct "Default" label (a binary indicator of who failed to pay back a loan), we must engineer a **proxy variable** to categorize users as high-risk (bad) or low-risk (good). We will utilize RFM (Recency, Frequency, Monetary) analysis to identify "disengaged" customers and label them as high-risk proxies.
* **Necessity:** Without historical loan performance data, behavioral data (transaction history) is the only available signal to estimate creditworthiness.
* **Business Risks:** The primary risk is **misclassification bias**. A customer might have low RFM scores simply because they are inactive, not necessarily because they are a credit defaulter. Treating low-activity users as "high risk" might exclude potential good borrowers (Type I error) or, conversely, highly active users might still default (Type II error).

**3. Trade-offs: Interpretable Models vs. High-Performance Models**
In the context of a regulated financial environment like Bati Bank, the trade-offs are distinct:
* **Logistic Regression with WoE (Weight of Evidence):**
    * **Pros:** Highly interpretable, produces a clear scorecard, and is the industry standard for regulatory compliance (easy to explain to auditors).
    * **Cons:** May capture fewer non-linear relationships, potentially resulting in lower predictive accuracy compared to complex ensembles.
* **Gradient Boosting (e.g., XGBoost/LightGBM):**
    * **Pros:** Typically offers higher predictive performance and accuracy by capturing complex, non-linear patterns in the data.
    * **Cons:** Acts as a "black box," making it difficult to explain individual decisions to regulators or customers. Requires additional tools (like SHAP) for explainability, which may not suffice for strict Basel II adherence.

***
