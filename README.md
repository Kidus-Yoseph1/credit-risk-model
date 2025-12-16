# Credit Risk Scorecard Model: An End-to-End ML Pipeline

This project details the development and deployment of a credit risk assessment system built on proprietary transaction data. The goal is to create a robust, auditable scorecard compliant with financial regulations using Logistic Regression and the Weight of Evidence (WoE) transformation technique.

---

## 1. Business Understanding & Regulatory Context

### **1.1. Influence of the Basel II Capital Accord on Model Interpretability**
The Basel II Capital Accord mandates that financial institutions maintain a rigorous framework for measuring and managing risk to ensure sufficient capital reserves. This regulatory requirement heavily influences our modeling approach by prioritizing **interpretability** and **auditability**. We must be able to explain *why* a specific risk score was assigned to a customer to comply with regulatory standards for transparency.

### **1.2. The Necessity and Risks of the Proxy Variable**
Since the provided dataset lacks a direct "Default" label, we utilize RFM (Recency, Frequency, Monetary) analysis to engineer a **proxy variable** to categorize users as high-risk or low-risk.

* **Business Risks:** The primary risk is **misclassification bias**, as low-activity users (low RFM) are treated as high risk, which may not correlate perfectly with credit default risk.

### **1.3. Trade-offs: Interpretable Models vs. High-Performance Models**
We prioritize **Logistic Regression** due to its superior interpretability and suitability for Basel II compliance, despite potential trade-offs in maximum predictive power compared to black-box models like XGBoost.

---

## 2. Local Setup and Execution Guide

Follow these steps to set up the project, train the model, and run the API locally.

### **2.1. Prerequisites**
* Python 3.9+
* Git
* Docker and Docker Compose (recommended for API deployment)

### **2.2. Environment Setup**
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kidus-Yoseph1/credit-risk-model.git
    cd credit-risk-model
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **2.3. Step-by-Step Pipeline Execution**

#### **A. Start MLflow Tracking Server**
The API and model training scripts require a local MLflow server running to register models and artifacts.

```bash
# Start the MLflow UI and server (keeps running in this terminal)
mlflow ui

```

(Access the UI at `http://localhost:5000`)

#### **B. Train and Register the Model**
Run the training script to perform feature engineering, model comparison, and register the best model (Logistic Regression) in the "Production" stage.

```bash
# Ensure you are in the project root directory
PYTHONPATH=. python src/model_train.py

```

#### **C. Run the API (Recommended: Docker Compose)**
Use Docker to run the FastAPI service, which loads the "Production" model from the local `mlruns.db` file.

1. **Build the image:**
```bash
docker-compose build

```


2. **Run the container:**
```bash
docker-compose up -d

```



The API will be available at `http://localhost:8000`.

### **2.4. Verification and Prediction**
Test the deployed API using the interactive documentation:

1. **Access Docs:** Open your browser to `http://localhost:8000/docs`.
2. **Test Prediction:** Use the `/predict` endpoint's "Try it out" feature with 6 WoE-transformed features (e.g., from the example in the report) to get a real-time risk probability.

---
