Task 5 – Credit Card Fraud Detection
📌 Description
This project focuses on detecting fraudulent credit card transactions using various machine learning algorithms. Since fraud data is rare compared to normal transactions, special attention is given to handling data imbalance.

📊 Dataset
The dataset is available on Kaggle here:

🔗 Click to Download Dataset from Kaggle

📝 Note: Due to large file size, the dataset is not included in this repository.
Please download it manually and place it in the project folder (e.g., dataset/ or data/).

⚙️ Features of the Dataset
284,807 transactions

492 fraud cases (highly imbalanced)

Features are anonymized (V1–V28), plus Time, Amount, and Class (0 = Normal, 1 = Fraud)

🛠️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

Matplotlib & Seaborn

🚀 Project Steps
Load and understand the dataset

Preprocess the data

Handle data imbalance (using techniques like SMOTE or RandomUnderSampler)

Train models: Logistic Regression, Decision Tree, Random Forest, etc.

Evaluate models with metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🧪 Output
Classification reports

Confusion matrix visualizations

Accuracy scores for different models

Graphs for fraud vs. normal transaction counts

▶️ How to Run
Clone this repo

Download the dataset from the Kaggle link above

Place the CSV in the dataset/ folder

Run the Python file or Jupyter Notebook
