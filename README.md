
# 🎓 Student Graduation Prediction using Machine Learning

This project aims to develop a predictive system that can classify whether a student will graduate or not based on academic exam scores and socio-demographic background. The project utilizes a public dataset and implements various machine learning models, ranging from baseline classifiers to tuned ensemble methods.

## 🧠 Problem Domain

Higher education plays a significant role in determining future success. However, not all students who enter college succeed in graduating on time—or at all. Early identification of at-risk students is essential to improve institutional effectiveness and provide timely intervention.

By leveraging structured data and machine learning, we aim to automatically and accurately predict graduation outcomes. This can help institutions make informed decisions and allocate resources more efficiently.

---

## 📊 Business Understanding

### ❓ Problem Statements

1. How can we predict student graduation based on exam scores and background information?
2. What are the key features that affect graduation outcomes?
3. Can predictive models assist institutions in designing early intervention strategies?

### 🎯 Project Goals

- Develop an accurate classification model to predict graduation status (`pass` or `fail`).
- Identify the most influential features that affect student graduation.
- Enable data-driven decision-making for academic interventions and student support systems.

### 💡 Solution Overview

The following machine learning workflow was adopted:

1. **Baseline Models**:  
   - Logistic Regression  
   - Decision Tree  

2. **Model Improvements with Tuning**:  
   - Random Forest (with GridSearchCV)  
   - XGBoost Classifier (with GridSearchCV)  

3. **Evaluation Metrics**:  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  

---

## 📂 Dataset Overview

- **Source**: [Students Performance in Exams - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size**: 1000 rows × 8 columns
- **Missing Values**: None
- **Features**:
  - Numerical: `math score`, `reading score`, `writing score`
  - Categorical: `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`

### 🔍 Feature Engineering

- `average_score`: average of math, reading, and writing scores
- `pass`: binary label (`1` if average score ≥ 70, else `0`)
- `score_level`: categorized into low (0–60), medium (60–80), high (80–100)

---

## 🧼 Data Preprocessing

1. **No missing values or duplicates**
2. **Outlier removal**: rows with `math score = 0` were dropped
3. **Encoding categorical variables** with `LabelEncoder`
4. **Train-test split** with stratification (80:20 ratio)
5. **Target distribution check** to ensure balanced classes

---

## 🤖 Modeling Results

| Model               | Accuracy | Precision/Recall | Notes                             |
|--------------------|----------|------------------|-----------------------------------|
| Logistic Regression| 97%      | Balanced         | Strong baseline                   |
| Decision Tree      | 96%      | Balanced         | Slightly lower than logistic      |
| Random Forest (Tuned) | 98%   | Balanced         | Excellent generalization          |
| XGBoost (Tuned)    | 98%      | Balanced         | **Best model overall**            |

✅ **XGBoost Classifier** was selected as the final model due to its high accuracy, interpretability, and stability.

---

## 📈 Evaluation

- **Confusion Matrix** (XGBoost):

  ```python
  [[107   1]
   [  3  89]]
  ```

- **Conclusion**:
  - The model achieved high accuracy and reliable classification performance.
  - Important predictive features include `math score`, `reading score`, `writing score`, `test preparation course`, and `parental education`.

---

## 💼 Business Impact

- 📌 **Early Intervention**: Institutions can identify and support at-risk students earlier.
- 📌 **Program Optimization**: `Test preparation courses` proved effective—institutions can scale them.
- 📌 **Resource Allocation**: Supports informed, data-driven academic decisions.

---

## 🧪 Tech Stack

- Python 3.10
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📁 File Structure

- `StudentsPerformance.csv`: Raw dataset
- `Submission_MLT_Hubbal.ipynb`: Notebook containing all steps from EDA to modeling
- `laporan_submission.md`: Documentation of project phases and findings
- `README.md`: Project summary and guide (this file)

---

## 📚 References

- [Students Performance in Exams Dataset - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- [OECD Education at a Glance 2023](https://www.oecd.org/education/education-at-a-glance/)
- [Predicting Student Academic Performance - ResearchGate](https://www.researchgate.net/publication/328110370_Predicting_Students_Academic_Performance_Using_Data_Mining_Techniques)
