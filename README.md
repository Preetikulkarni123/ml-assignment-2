# Machine Learning Assignment 2
## BITS Pilani - M.Tech (AIML/DSE)

---

## Problem Statement

This project aims to predict whether a patient has heart disease based on 13 clinical features using multiple machine learning classification algorithms. The objective is to evaluate the performance of six different machine learning models using comprehensive evaluation metrics, deploy an interactive web application using Streamlit, and demonstrate the complete end-to-end ML workflow from data preprocessing to model deployment.

**Goals:**
- Implement 6 classification algorithms (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost)
- Evaluate models using 6 key metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Build an interactive Streamlit web application
- Deploy the application on Streamlit Community Cloud
- Compare model performance and provide insights

---

## Dataset Description

**Dataset Name:** Heart Disease Dataset

**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/heart+Disease

**Description:**
This dataset contains clinical measurements from patients to predict the presence of heart disease. It includes various medical attributes such as age, sex, chest pain type, blood pressure, cholesterol levels, and other cardiac-related measurements.

**Dataset Characteristics:**
- **Total Features:** 13 features (all numerical)
- **Target Variable:** target (Binary classification: 0 = No Disease, 1 = Disease)
- **Total Instances:** 297 rows (after removing missing values)
- **Missing Values:** Removed during preprocessing
- **Class Distribution:** 
  - Class 0 (No Disease): 160 instances (53.9%)
  - Class 1 (Disease): 137 instances (46.1%)

**Key Features:**
1. **age:** Age in years
2. **sex:** Sex (1 = male, 0 = female)
3. **cp:** Chest pain type (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps:** Resting blood pressure (mm Hg on admission)
5. **chol:** Serum cholesterol (mg/dl)
6. **fbs:** Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg:** Resting electrocardiographic results (0-2)
8. **thalach:** Maximum heart rate achieved
9. **exang:** Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak:** ST depression induced by exercise relative to rest
11. **slope:** Slope of the peak exercise ST segment (0-2)
12. **ca:** Number of major vessels colored by fluoroscopy (0-3)
13. **thal:** Thalassemia type (0-3)

**Data Preprocessing:**
- Handled missing values by removal
- All features are numerical (no categorical encoding needed)
- Scaled numerical features using StandardScaler
- Train-test split: 80-20 ratio (237 train, 60 test) with stratification to maintain class balance

---

## Models Used

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8333 | 0.9498 | 0.8462 | 0.7857 | 0.8148 | 0.6652 |
| Decision Tree | 0.6833 | 0.6786 | 0.6800 | 0.6071 | 0.6415 | 0.3614 |
| kNN | 0.8833 | 0.9492 | 0.9200 | 0.8214 | 0.8679 | 0.7680 |
| Naive Bayes | 0.8833 | 0.9375 | 0.8889 | 0.8571 | 0.8727 | 0.7655 |
| Random Forest (Ensemble) | 0.8667 | 0.9420 | 0.8846 | 0.8214 | 0.8519 | 0.7326 |
| XGBoost (Ensemble) | 0.8333 | 0.9051 | 0.8750 | 0.7500 | 0.8077 | 0.6683 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved 83.33% accuracy with excellent AUC score of 0.9498, indicating strong ability to distinguish between classes. Good baseline performance with balanced precision (0.8462) and recall (0.7857). Fast training time and interpretable coefficients make it suitable for understanding feature importance. Works well for this medical dataset where linear relationships exist between features and target. |
| Decision Tree | Lowest performance among all models with only 68.33% accuracy and poor AUC (0.6786). Shows clear signs of overfitting on training data with high variance. The model creates rigid decision boundaries that don't generalize well to test data. Easy to visualize and interpret but requires careful pruning and regularization. Not recommended for this dataset. |
| kNN | **Best performing model** with 88.33% accuracy and excellent AUC of 0.9492. Achieved highest precision (0.92) and strong MCC score (0.7680). The model benefits from StandardScaler preprocessing as it is distance-based. With k=5, it effectively captures local patterns in the data. However, prediction time is slower compared to other models as it requires distance calculation to all training points. |
| Naive Bayes | Tied for best accuracy (88.33%) with outstanding performance across all metrics. Surprisingly effective despite assuming feature independence. Achieved excellent AUC (0.9375) and balanced precision-recall (0.8889/0.8571). Fastest training and prediction time among all models. The Gaussian assumption works well for continuous medical features. Highly recommended for real-time prediction scenarios. |
| Random Forest (Ensemble) | Strong performance with 86.67% accuracy and high AUC (0.9420). Ensemble averaging of 100 decision trees effectively reduces overfitting seen in single decision tree. Provides robust predictions with good balance between bias and variance. Feature importance scores available for medical interpretation. Slightly slower than simpler models but provides reliable predictions. |
| XGBoost (Ensemble) | Achieved 83.33% accuracy with good AUC (0.9051). Sequential boosting approach shows strong precision (0.875) but lower recall (0.75), indicating conservative predictions. Handles class imbalance well and provides built-in regularization. Training time is longer than other models but offers advanced features like handling missing values. With proper hyperparameter tuning, could potentially achieve better performance. |

**Key Insights:**
- **Best Overall Models:** kNN and Naive Bayes (tied at 88.33% accuracy)
- **Worst Model:** Decision Tree (68.33% accuracy) - shows clear overfitting
- **Best AUC Score:** Logistic Regression (0.9498) - excellent discrimination ability
- **Best Precision:** kNN (0.92) - fewer false positives
- **Best Recall:** Naive Bayes (0.8571) - better at catching actual disease cases
- **Trade-offs:** 
  - kNN: High accuracy but slow prediction time
  - Naive Bayes: Fast and accurate but assumes feature independence
  - Random Forest: Balanced performance with interpretability through feature importance
  - XGBoost: Good performance but requires more computational resources

---

## Project Structure

```
ml-assignment-2/
│
├── complete_solution.py            # Complete training script
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── models/                         # Saved trained models
│   ├── model_lr.pkl               # Logistic Regression model
│   ├── model_dt.pkl               # Decision Tree model
│   ├── model_knn.pkl              # K-Nearest Neighbors model
│   ├── model_nb.pkl               # Naive Bayes model
│   ├── model_rf.pkl               # Random Forest model
│   ├── model_xgb.pkl              # XGBoost model
│   └── scaler.pkl                 # StandardScaler for preprocessing
│
├── heart_disease.csv          # Full dataset (297 rows)
└── sample_test_data.csv       # Sample test data (20 rows)
│                     
|── model_results.csv          # Performance metrics
└── model_observations.csv     # Model observations
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git



## Evaluation Metrics Explained

1. **Accuracy:** Proportion of correct predictions out of total predictions
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Best: kNN and Naive Bayes (88.33%)

2. **AUC (Area Under ROC Curve):** Measures model's ability to distinguish between classes
   - Range: 0 to 1 (higher is better)
   - Best: Logistic Regression (0.9498)

3. **Precision:** Proportion of true positives among predicted positives
   - Formula: TP / (TP + FP)
   - Best: kNN (0.9200)

4. **Recall (Sensitivity):** Proportion of true positives among actual positives
   - Formula: TP / (TP + FN)
   - Best: Naive Bayes (0.8571)

5. **F1 Score:** Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Best: Naive Bayes (0.8727)

6. **MCC (Matthews Correlation Coefficient):** Balanced measure even for imbalanced datasets
   - Range: -1 to +1 (higher is better)
   - Best: kNN (0.7680)

---

## Technology Stack

- **Programming Language:** Python 3.8+
- **Web Framework:** Streamlit
- **ML Libraries:** 
  - scikit-learn 1.3.0 (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest)
  - XGBoost 2.0.0 (Gradient Boosting)
- **Data Processing:** Pandas 2.0.3, NumPy 1.24.3
- **Visualization:** Matplotlib 3.7.2, Seaborn 0.12.2
- **Model Persistence:** Joblib 1.3.2

---

## Results Summary

### Best Performing Model
- **Model:** kNN (K-Nearest Neighbors)
- **Accuracy:** 0.8833 (88.33%)
- **AUC:** 0.9492
- **F1 Score:** 0.8679
- **Key Strength:** Highest precision (0.92) and strong overall performance

### Runner-up Model
- **Model:** Naive Bayes
- **Accuracy:** 0.8833 (88.33%)
- **AUC:** 0.9375
- **F1 Score:** 0.8727
- **Key Strength:** Fastest training/prediction time with excellent recall

### Most Interpretable Model
- **Model:** Logistic Regression
- **Accuracy:** 0.8333 (83.33%)
- **AUC:** 0.9498 (Best AUC score)
- **Reason:** Provides interpretable coefficients showing feature importance

### Training Statistics
- **Training Set Size:** 237 samples (80%)
- **Test Set Size:** 60 samples (20%)
- **Total Training Time:** ~3 minutes (all 6 models)
- **Feature Scaling:** StandardScaler applied

---

## Author

**Name:** Preeti Kulkarni  
**Student ID:** 2025AA05894
**Course:** M.Tech (AIML/DSE)  
**Institute:** BITS Pilani  
**Assignment:** Machine Learning - Assignment 2  
**Date:** 14 February 2026  
**Email:** 2025aa05894@wilp.bits-pilani.ac.in
**GitHub:** Preetikulkarni123

---



### Dependencies Version
```
streamlit==1.28.0
scikit-learn==1.3.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
xgboost==2.0.0
joblib==1.3.2
```
