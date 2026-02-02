# Credit Card Fraud Detection

## Project Overview

This project implements a **robust credit card fraud detection system** designed to handle **real-world challenges** in financial AI. Fraud detection is inherently difficult due to **extreme class imbalance**, evolving fraud patterns, and the high cost of misclassification. This system leverages **machine learning models** (Logistic Regression, XGBoost) to maximize fraud detection while minimizing financial losses.

---

## The Challenge

Credit card fraud detection presents several critical challenges:

1. **Severely Imbalanced Data**  
   - Fraud transactions make up **less than 0.2%** of all records.  
   - Naive models may appear highly accurate while **failing to detect actual fraud**.  

2. **Asymmetric Error Costs**  
   - **False negatives (missed fraud)** → direct financial loss.  
   - **False positives (blocking legitimate transactions)** → damages customer trust.  
   - The model must **balance recall and precision** to minimize overall business cost.

3. **Generalization & Data Leakage**  
   - The model must **generalize to unseen transactions**.  
   - Avoiding **data leakage** is crucial to prevent artificially inflated performance metrics.

> Real-world financial AI systems face **rare labels, evolving patterns, and high-stakes decisions**. The focus is on **business impact and explainability**, not just raw accuracy.

---

## Project Goals

- Detect fraudulent transactions with **high recall**.  
- Minimize false alarms to protect legitimate customers.  
- Calculate **business cost** of misclassifications for informed decision-making.  
- Build a pipeline that scales to **unseen data** without leakage.  

---

## Dataset

- Source: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Features: PCA-transformed features `V1–V28`, `Time`, `Amount`  
- Target: `Class` (0 = Legitimate, 1 = Fraud)  
- Imbalance: Fraud transactions < 0.2% of total

---

## Features & Methodology

- **Preprocessing:**  
  - Standard scaling for `Time` and `Amount`  
  - Stratified train-test split to maintain class distribution

- **Models:**  
  - Logistic Regression with `class_weight` adjustment  
  - XGBoost with `scale_pos_weight` for imbalance  

- **Evaluation Metrics:**  
  - Confusion matrix  
  - Accuracy, Precision, Recall, F1-score  
  - Precision-Recall AUC (PR-AUC)  
  - Business cost calculation using **COST_FN** and **COST_FP**  

- **Threshold Tuning:**  
  - Business-optimized probability threshold (e.g., 0.2–0.3) to maximize recall and minimize financial loss

---

## Model Performance

### Logistic Regression & XGBoost

```text
Confusion Matrix:
[[51333  5531]
 [    6    92]]

Accuracy          : 0.9028
Recall (Fraud)    : 0.9388
Precision (Fraud) : 0.0164
F1-Score (Fraud)  : 0.0322
PR-AUC            : 0.7190

Confusion Matrix: (XGBoost)
[[56851    13]
 [   14    84]]

Accuracy          : 0.9995
Recall (Fraud)    : 0.8571
Precision (Fraud) : 0.8660
F1-Score (Fraud)  : 0.8615
PR-AUC            : 0.8816
```
---

### Clone the Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git


