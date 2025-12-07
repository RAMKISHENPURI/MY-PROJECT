# Dynamic Pricing for Airbnb Hosts in Bristol

<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/4eedc339-8e49-42b5-b8eb-8f19885b42d4" />

This repository showcases an end-to-end AI-driven pricing framework designed to help small Airbnb hosts in Bristol optimise their revenue.
The solution integrates machine learning, explainable AI, financial simulation, and dashboard visualisation to support data-driven pricing decisions.

It demonstrates how analytics can move beyond prediction to economic value creation.

**Live Interactive Dashboard:** [**https://bristol-airbnb-dashboard.onrender.com/**](https://bristol-airbnb-dashboard.onrender.com/)

---

## Key Results

This section quantifies the model’s technical performance, interpretability insights, and economic value.

✔ Model Performance:
The predictive model achieved a 0.7884 AUC-ROC, demonstrating reliable booking probability forecasting.

✔ Explainability Findings:
SHAP analysis identified review recency, month, and price as the top demand drivers, offering interpretable insights for non-technical decision-makers.

✔ Business Impact:
Financial simulation revealed:

📌 41.82% uplift vs. host-listed pricing  
📌 24.37% uplift vs. static average pricing

➡ These outcomes highlight the commercial value of data-driven dynamic pricing for small Airbnb hosts.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RAMKISHENPURI/MY-PROJECT.git
    cd MY-PROJECT
    ```

2.  **Install libraries:**
    ```bash
    pip install pandas numpy scikit-learn xgboost catboost shap matplotlib seaborn
    ```

3.  **Run the analysis:**
    Launch Jupyter and run the notebooks in the following order:
    1.  `FINAL PROJECT.ipynb` (Data Cleaning & Preparation)
    2.  `EDA.ipynb` (Exploratory Data Analysis)
    3.  `MODELLING.ipynb` (Model Training & Evaluation)
    4.  `ANALYSIS.ipynb` (Financial Simulation & Interpretation)
    5.  `DASHBOARD.ipynb` (Interactive Dashboard)

---

## Core Technologies

* **Modeling:** XGBoost, CatBoost, Scikit-learn
* **Interpretability:** SHAP
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
