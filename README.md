# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning project that detects **fraudulent credit card transactions** from a highly imbalanced real-world dataset. Built with Python in a Jupyter Notebook, it trains and compares multiple classifiers and evaluates them using industry-standard fraud detection metrics.

---

## ✨ Features

- 🔍 Detects fraudulent transactions from 284,807 real credit card records
- ⚖️ Handles severe class imbalance using **undersampling** techniques
- 📊 Full **Exploratory Data Analysis (EDA)** with transaction pattern visualizations
- 🧠 Trains and compares three ML classifiers — Logistic Regression, Decision Tree, and Random Forest
- 📈 Evaluates models using Confusion Matrix, Precision, Recall, F1 Score, and ROC-AUC Curve
- 💾 Pre-trained model saved as `credit_card_model.pkl` for direct reuse

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Notebook | Jupyter Notebook |
| ML Library | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Pickle (`.pkl`) |

---

## 🧪 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Origin:** Real transactions by European cardholders, September 2013
- **Size:** 284,807 transactions — only **492 are fraud (0.17%)**
- **Features:** 28 PCA-anonymized features (`V1`–`V28`) + `Time`, `Amount`, and `Class`
  - `Class = 0` → Legitimate
  - `Class = 1` → Fraudulent

> ⚠️ The dataset is highly imbalanced. Undersampling is applied to balance the classes before training.

---

## ⚙️ How It Works

```
Raw Transaction Data (284,807 records)
          ↓
Exploratory Data Analysis (EDA)
(distribution plots, correlation heatmap, fraud vs. legit analysis)
          ↓
Handle Class Imbalance → Undersampling
          ↓
Feature Scaling (StandardScaler on Amount & Time)
          ↓
  ┌─────────────────────────────────┐
  │  Logistic Regression            │
  │  Decision Tree Classifier       │
  │  Random Forest Classifier ✅    │  ← Best Model
  └─────────────────────────────────┘
          ↓
Evaluation: Confusion Matrix | Precision | Recall | F1 | ROC-AUC
```

---

## 📊 Model Performance

| Model | ROC-AUC Score |
|---|---|
| Logistic Regression | ~0.97 |
| Decision Tree | ~0.91 |
| **Random Forest** | **~0.93** |

> Best overall balance of precision and recall achieved by **Random Forest Classifier**.

---

## 📂 Project Structure

```
📦 CREDIT-CARD-FRAUD-DETECTION-USING-ML
├── CREADIT CARD FRAUD DETECTION.ipynb    # Main notebook
├── credit_card_model.pkl                 # Pre-trained Random Forest model
└── README.md
```

> 📌 The dataset (`creditcard.csv`) is not included due to size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook

1. **Clone the repository**
   ```bash
   git clone https://github.com/RUWAKAN/CREDIT-CARD-FRAUD-DETECTION-USING-ML.git
   cd CREDIT-CARD-FRAUD-DETECTION-USING-ML
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the root folder.

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open** `CREADIT CARD FRAUD DETECTION.ipynb` and run all cells.

---

## 🔁 Using the Pre-trained Model

```python
import pickle
import numpy as np

# Load the model
with open('credit_card_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example transaction (28 PCA features + Time + Amount)
transaction = np.array([[0.0, -1.35, 1.19, ..., 0.01, 149.62]])

prediction = model.predict(transaction)
print("FRAUD" if prediction[0] == 1 else "LEGITIMATE")
```

---

## 🗺️ Roadmap

- [ ] Add SMOTE oversampling for better fraud recall
- [ ] Benchmark XGBoost and LightGBM models
- [ ] Build a real-time fraud detection API with Flask
- [ ] Deploy as a web app using Streamlit
- [ ] Add SHAP explainability for model transparency

---

## 👨‍💻 Author

**Ashish Kumar (RUWAKAN)**
BCA (2022–2025) — Galgotias University

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Free to use and modify for learning and research purposes.
