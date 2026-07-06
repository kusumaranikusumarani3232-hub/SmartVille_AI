# 🏙️ SmartVille AI — Civic Intelligence Platform

SmartVille AI is a state-of-the-art civic intelligence platform built with **Streamlit**, **scikit-learn**, **Plotly**, and **Google Gemini**. It enables local citizens and city administrators to monitor environmental health, report civic complaints, analyze trends using machine learning, and interact with data through an AI assistant.

Developed for the **Google Cloud AI Hackathon 2024**.

---

## 🚀 Key Features

### 1. 📊 Community Dashboard
An at-a-glance overview of SmartVille's health across all districts.
* **Key Performance Indicators (KPIs)**: Track total complaints, average AQI, complaint resolution rates, and active critical issues.
* **Interactive Trends**: Visualise weekly complaints by category and rolling AQI averages.
* **District Performance**: Side-by-side district comparison charts (average AQI, total complaints, and resolution percentages).
* **Alerts Feed**: Live indicators of environmental anomalies (e.g. hazardous AQI, noise peaks, poor water quality) and open critical citizen complaints.

### 2. 🌬️ Environmental Analytics
Deep-dive environmental analytics with local machine learning models.
* **Environmental Metrics**: Detailed logs of Air Quality Index (AQI), PM2.5, PM10, Noise (dB), and Water Quality Index (WQI).
* **ML-powered AQI Predictor**: Uses a `RandomForestRegressor` to predict AQI based on particulate matter levels, current weather, month, and district.
* **Explainable AI (SHAP)**: Explains individual machine learning predictions using Plotly-based SHAP waterfall charts, showing how each feature contributed to the predicted AQI.
* **Anomaly Detection**: Flags dates when AQI, noise, or water quality exceeds standard safety thresholds.

### 3. 📢 Citizen Complaint Intelligence
A dual-view workspace tailored for both residents and municipal managers:
* **Citizen View**:
  * **Submit a Complaint**: Report civic issues (Roads, Water, Garbage, Electricity, etc.) with automated AI priority classification.
  * **Track Complaints**: Real-time status lookup using generated ticket tracking IDs.
  * **Local Feed**: Interactive card list showcasing recent neighborhood reports where citizens can upvote issues to signal urgency.
* **Administrator View**:
  * **KPI Board & Category Charts**: Distribution maps and charts categorizing issues by district, category, and current status.
  * **Geographic Map**: Dynamic scatter map locating complaints with priority color coding.
  * **Resolution Tracker**: Average resolution times per category to highlight bottlenecks.

### 4. 🤖 AI Assistant (Powered by Gemini)
An intelligent chatbot that integrates with the platform's data engine.
* **Context-Aware Analytics**: Automatically parses the current filtered dataset (district, date ranges) into a compressed summary.
* **Zero-Knowledge Computations**: The local platform handles calculations while Gemini translates the metrics into concise, actionable summaries.
* **Secure API Configuration**: Set your API key directly in the application sidebar or Streamlit secrets for secure operation.

---

## 🛠️ Architecture & Tech Stack

* **Frontend**: Streamlit (configured with custom CSS, glassmorphism UI, and dark mode theme)
* **Visualizations**: Interactive Plotly plots and maps
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: 
  * `scikit-learn` (Random Forest regressor and classifier pipelines)
  * `shap` (TreeExplainer waterfall plots)
* **Generative AI**: Google Gemini API (`gemini-1.5-flash`)
* **Database**: SQLite (persisting citizen reports locally)

---

## 📦 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your machine.

### 2. Clone and Setup Environment
Navigate to your project directory and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Running the App
Start the Streamlit application:
```bash
streamlit run app.py
```
On first startup, the application will:
1. Generate synthetic environmental and complaint datasets (`.csv` files under `data/`).
2. Train the AQI and priority classifier models and save them (`.joblib` files under `ml/models/`).
3. Launch the server (default: `http://localhost:8501`).

### 4. Enable AI Assistant
* Obtain a free API key from [Google AI Studio](https://aistudio.google.com).
* Paste it into the **Gemini API Key** field in the application sidebar.
* Alternatively, add it to your secrets file at `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your-api-key-here"
  ```
