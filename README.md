# Automated Student At-Risk Detection Dashboard

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E.svg)
![XGBoost](https://img.shields.io/badge/xgboost-3.4.1-13A6E0.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.49.1-8D45E8.svg)

## 📌 Project Overview & System Architecture

This project is a modular, end-to-end Machine Learning web application designed to predict the probability that a student is "at-risk" based on their demographic and academic profile.

**System Architecture:**
1. **Interactive UI (Streamlit):** A graphical dashboard allowing users to input student demographics and exact test scores via dropdowns and sliders.
2. **Data Ingestion & Transformation:** The pipeline automatically scales features using `StandardScaler` and encodes categorical variables with `OneHotEncoder`.
3. **Machine Learning Model:** A highly accurate **Voting Ensemble** combining **XGBoost** and **Random Forest** algorithms. 
4. **Model Explainability (SHAP):** Every prediction is passed through a SHAP `KernelExplainer` to provide transparent, instance-level feature importance, rendering a dynamic feature contribution plot directly in the UI explaining *why* the model made its decision.

---

## 🛠️ Setup & Local Execution Instructions

Follow these steps to run the dashboard locally on your machine.

**1. Clone the repository and navigate into it:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create and activate a virtual environment:**
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

*(If you wish to retrain the models from scratch, run `python -m src.pipeline.train_pipeline`)*

**4. Start the Streamlit application:**
```bash
streamlit run app.py
```
*Your browser will automatically open the interactive dashboard at `http://localhost:8501`.*

---

## ☁️ Deployment (Streamlit Community Cloud)

This project is fully configured to be deployed for free natively on **Streamlit Community Cloud**.

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, set the main file path to `app.py`, and click **Deploy**!
4. Streamlit will automatically read your `requirements.txt` and host the application for you.
