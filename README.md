# CREDIT-CARD-FRAUD-DETECTION-USING-ML

💳 Credit Card Fraud Detection using Machine Learning
Author: Ashish Kumar
Project Type: Academic / ML Application
Technologies Used: Python, Scikit-learn, Pandas, Matplotlib, Seaborn



🔍 Overview
This project aims to detect fraudulent credit card transactions using machine learning algorithms. With imbalanced data being a key challenge, the solution focuses on techniques to handle skewed datasets and improve the detection of rare fraud cases.

The system uses historical transaction data to train classifiers and identify patterns that suggest fraudulent behavior.



🚀 Features
⚖️ Imbalanced Data Handling: Utilizes undersampling techniques.

📊 Exploratory Data Analysis (EDA): Insightful visualizations for understanding transaction patterns.



🧠 ML Algorithms:

Logistic Regression

Decision Tree

Random Forest



📈 Model Evaluation Metrics:

Confusion Matrix

Accuracy, Precision, Recall, F1 Score

ROC-AUC Curve


jupyter notebook "Credit Card Fraud Detection (1).ipynb"


📁 Dataset

Source: Kaggle — Credit Card Fraud Detection Dataset

Content: Transactions made by European cardholders in September 2013.

Size: 284,807 transactions with 492 frauds (highly imbalanced).

Features: PCA-transformed anonymized features + Time, Amount, and Class (0 = non-fraud, 1 = fraud)



📉 Model Performance
Best performing model: Random Forest Classifier

ROC-AUC Score: ~0.93

Metrics visualized via ROC Curve and confusion matrices.


📌 Future Work
Incorporate oversampling (SMOTE) or ensemble methods (XGBoost, LightGBM)

Real-time fraud detection system via API

Model deployment using Flask or Streamlit


📁 Project Structure

📦 credit-card-fraud-detection
├── Credit Card Fraud Detection (1).ipynb
├── requirements.txt
├── README.md
└── assets/ (e.g. ROC curves, confusion matrix plots)



🧑‍💻 Author
Ashish Kumar
BCA (2022-2025)
Galgotias University

📜 License
This project is licensed under the MIT License. Free to use and modify for learning and research purposes.
