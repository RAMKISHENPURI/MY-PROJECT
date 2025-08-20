# Dynamic Pricing Framework for Airbnb

This repository contains a machine learning framework to provide dynamic pricing recommendations for Airbnb hosts, using a case study of Bristol, UK. All required datasets and Jupyter Notebooks are included. The goal is to help small-scale hosts optimize revenue by forecasting daily booking probability.

**Live Demo:** [**https://bristol-airbnb-dashboard.onrender.com/**](https://bristol-airbnb-dashboard.onrender.com/)

---

## Key Result

The financial simulation showed that this dynamic pricing framework could increase host revenue by **41.82%** compared to their original listed prices.

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
