Credit Card Fraud Detection using Machine Learning

An End-to-End Classification Project

Created by: Antara & Aditya
Organization: NeuroXSentinel

Project Overview

Credit card fraud detection is a real-world machine learning problem involving highly imbalanced data and critical decision-making.
This project demonstrates a complete end-to-end machine learning pipeline, starting from raw data understanding to applying and comparing multiple classification algorithms.

The goal is to accurately identify fraudulent transactions while minimizing false positives.

🎯 Objectives

Understand and prepare real-world financial transaction data

Handle class imbalance effectively

Apply multiple classification algorithms

Evaluate models using industry-standard metrics

Build a project suitable for real-world deployment and learning

🧠 Machine Learning Concepts Covered

Binary Classification

Train–Test Split with Stratification

Feature Scaling (StandardScaler)

Handling Imbalanced Datasets

Model Evaluation & Comparison

🗂 Dataset Description

Dataset Name: Credit Card Transactions

Target Column: Class

0 → Legitimate Transaction

1 → Fraudulent Transaction

Dataset is highly imbalanced, reflecting real-world scenarios.

⚙️ Models Implemented

The following classification algorithms were implemented and compared:

Logistic Regression (Baseline Model)

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Support Vector Machine (SVM)

Random Forest Classifier

Each model was evaluated using consistent metrics for fair comparison.

📊 Evaluation Metrics

Precision

Recall

F1-Score

Confusion Matrix

ROC-AUC Score

ROC Curve Comparison

Accuracy alone was avoided due to class imbalance.

🛠️ Tech Stack

Language: Python

Libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

📁 Project Workflow

Data Loading & Understanding

Exploratory Data Analysis (EDA)

Feature & Target Separation

Train-Test Split with Stratification

Feature Scaling

Model Training

Model Evaluation

Model Comparison & Selection

📌 Key Learnings

Importance of stratified sampling in imbalanced datasets

Why precision and recall matter more than accuracy

How different classifiers behave on real-world financial data

End-to-end ML project structuring for production readiness

🚀 Future Enhancements

SMOTE for advanced imbalance handling

Hyperparameter tuning using GridSearchCV

XGBoost / LightGBM implementation

Model deployment using Flask / FastAPI

🤝 Contributors

Antara

Aditya

Organization: NeuroX Sentinel

⭐ If you find this project helpful

Give it a ⭐ and share it with fellow learners!
