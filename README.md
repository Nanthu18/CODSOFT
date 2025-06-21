# 🔐 Credit Card Fraud Detection - Task 5

This project focuses on detecting fraudulent credit card transactions using machine learning. The goal is to build a model that can classify transactions as **genuine** or **fraudulent**, and help prevent financial losses due to fraud.

---

## 📁 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗 [Click here to download the dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> After downloading, place the `creditcard.csv` file in the same directory as your Python code.

---

## 🛠️ Technologies & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)

---

## 🔄 Project Workflow

1. **Data Preprocessing**  
   - Removed `Time` column  
   - Scaled `Amount` using StandardScaler

2. **Class Balancing**  
   - Used **SMOTE** to handle class imbalance (fraud cases are very rare)

3. **Model Training**  
   - Trained a **Random Forest Classifier** (fast & accurate)

4. **Model Evaluation**  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
   - Top 10 Important Features  
   - Visualizations

---

## 📊 Output Charts

- Class Distribution (Genuine vs Fraud)
- Confusion Matrix
- Feature Importance
- Fraud vs Amount (Boxplot)

---

## ✅ Conclusion

The final model is able to detect fraudulent transactions with good precision and recall after handling class imbalance. The project demonstrates a complete fraud detection pipeline, from raw data to evaluation.

---

## 💬 Author

**Nanthini Senthil Murugan**  
CodSoft Internship – Machine Learning Track (Task 5)

