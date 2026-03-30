# 🎓 Student Performance Anomaly Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-anomaly-detection-a4bpewzbdrxkymhpcz7mww.streamlit.app)

A web‑based dashboard that helps educators identify at‑risk students earlier by combining **predictive modeling** and **anomaly detection**, with clear explanations of why each student was flagged.

**Live Demo:** [https://student-performance-anomaly-detection-a4bpewzbdrxkymhpcz7mww.streamlit.app](https://student-performance-anomaly-detection-a4bpewzbdrxkymhpcz7mww.streamlit.app)

---

## 📌 Features

- **Predictive Analytics** – Random Forest and XGBoost models forecast student outcomes based on historical patterns.
- **Anomaly Detection** – Isolation Forest and DBSCAN spot unusual behaviour (e.g., sudden drop in login frequency, irregular submissions).
- **Interactive Dashboard** – Explore student profiles, cohort analytics, and risk trends with Plotly charts.
- **Interpretable Explanations** – Each risk flag comes with a natural language explanation and recommended actions.
- **Open Data** – Uses the publicly available OULAD and UCI Student Performance datasets.

---

## 🛠️ Tech Stack

- **Python** 3.10+
- **Streamlit** – Dashboard framework
- **pandas**, **numpy** – Data processing
- **scikit-learn**, **xgboost** – Machine learning
- **plotly**, **matplotlib**, **seaborn** – Visualizations
- **SQLite** – Lightweight database for storing results

---

## 🚀 Getting Started (Local Run)

### 1. Clone the repository

```bash
git clone https://github.com/Emmy-Okello2030/student-performance-anomaly-detection.git
cd student-performance-anomaly-detection
