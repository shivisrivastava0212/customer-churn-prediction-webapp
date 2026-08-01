# 🔮 Customer Churn Prediction Web Application

An end-to-end Machine Learning project that predicts the probability of telecommunication customer churn. This project covers the complete ML pipeline—from exploratory data analysis and model training in Google Colab to building an interactive Streamlit web application and deploying it on Streamlit Community Cloud.

🚀 **Live Demo:** https://shivi-telco-churn.streamlit.app

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Feature Importance (Top Drivers)](#-feature-importance-top-drivers)
- [How to Run Locally & in Google Colab](#-how-to-run-locally--in-google-colab)
- [Deployment Instructions](#-deployment-instructions)
- [Repository Structure](#-repository-structure)

---

# 📊 Project Overview

Customer churn occurs when customers discontinue their services with a company. Predicting churn enables telecom providers to identify high-risk customers and take proactive retention measures.

This web application allows users to enter customer information such as demographics, account details, contract information, and billing data to instantly predict the probability of customer churn using a trained Machine Learning model.

## ✨ Key Features

- 🎯 Interactive Streamlit user interface
- ⚡ Instant real-time churn prediction
- 📈 Displays churn probability percentage
- 🟢🟡🔴 Color-coded risk assessment
- 🤖 Production-ready trained ML model
- ☁️ Deployed on Streamlit Community Cloud

---

# 🏗️ System Architecture

```text
                Telco Customer Dataset
                         │
                         ▼
             Data Cleaning & Preprocessing
                         │
                         ▼
               Feature Engineering
                         │
                         ▼
          Machine Learning Model Training
                         │
                         ▼
              Save Model (best_model.pkl)
                         │
                         ▼
               Streamlit Web Application
                         │
                         ▼
                User Input Prediction
                         │
                         ▼
           Churn Probability & Risk Level
```

---

# 🧠 Model Performance & Evaluation

The production model evaluates customer profiles by considering overall performance as well as class-specific metrics.

> **Production Model:** `best_model.pkl`

## 🎯 Overall Accuracy

**79.77%**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|--------|----------:|--------:|----------:|--------:|
| **0 (Will Stay)** | 0.84 | 0.89 | 0.86 | 1009 |
| **1 (Will Churn)** | 0.67 | 0.56 | 0.61 | 400 |
| **Macro Average** | 0.75 | 0.73 | 0.74 | 1409 |
| **Weighted Average** | 0.79 | 0.80 | 0.79 | 1409 |

---

# 📈 Feature Importance (Top Drivers)

The trained model identifies several influential features affecting customer churn.

## 📌 Positive Drivers (Increase Churn Risk)

- Monthly Charges (~0.70)
- Total Charges (~0.50)
- Paperless Billing

Higher monthly expenses significantly increase the likelihood of customer churn.

## 📌 Negative Drivers (Decrease Churn Risk)

- Dependents (~-0.60)
- Phone Service (~-0.25)

Customers with dependents or bundled phone services are generally less likely to churn.

---

# 💻 How to Run Locally & in Google Colab

## Option A — Run the Training Notebook in Google Colab

If you want to retrain or explore the machine learning model:

1. Open **Google Colab**
2. Click **File → Upload Notebook**
3. Upload the training notebook (`churn_training.ipynb`)
4. Upload the dataset (`Telco_customer_churn.xlsx`)
5. Click **Runtime → Run all**
6. The notebook will:
   - Clean the data
   - Perform preprocessing
   - Train the model
   - Evaluate performance
   - Save the trained model as `best_model.pkl`

---

## Option B — Run the Streamlit App Locally

### 1. Clone the repository

```bash
git clone https://github.com/shivirrivastava0212/customer-churn-prediction-webapp.git

cd customer-churn-prediction-webapp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit application

```bash
streamlit run app.py
```

### 4. Open the application

```
http://localhost:8501
```

---

# 🌐 Deployment Instructions

This project is deployed using **Streamlit Community Cloud**.

## Steps

1. Push your project to GitHub.

2. Open:

```
https://share.streamlit.io
```

3. Click **Deploy an App**.

4. Configure the following:

| Setting | Value |
|----------|-------|
| Repository | `shivirrivastava0212/customer-churn-prediction-webapp` |
| Branch | `main` |
| Main file | `app.py` |
| App URL | `shivi-telco-churn` |

### Recommended Python Version

Use **Python 3.12** for smoother package installation and better compatibility with scientific libraries such as `contourpy`.

5. Click **Deploy**.

---

# 📂 Repository Structure

```text
customer-churn-prediction-webapp/
│
├── app.py
├── best_model.pkl
├── churn_training.ipynb
├── requirements.txt
├── README.md
└── Telco_customer_churn.xlsx
```

### File Description

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `best_model.pkl` | Trained machine learning model |
| `churn_training.ipynb` | Google Colab/Jupyter notebook |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |
| `Telco_customer_churn.xlsx` | Dataset used for training |

---

# 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Joblib
- Google Colab
- Git
- GitHub
- Streamlit Community Cloud

---

# 📌 Future Improvements

- Hyperparameter optimization
- Ensemble learning methods
- Explainable AI using SHAP/LIME
- Customer retention recommendations
- Docker containerization
- REST API integration
- CI/CD pipeline
- Cloud deployment on AWS or Azure

---

# 👩‍💻 Author

**Shivi Srivastava**

- B.Tech Computer Science Engineering
- Machine Learning & AI Enthusiast

---

## ⭐ If you found this project helpful, consider giving it a Star on GitHub!
