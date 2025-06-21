# 📌 Credit Card Fraud Detection 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Step 1: Load data
df = pd.read_csv('creditcard.csv')
print("🧾 Total transactions:", df.shape[0])

# Step 2: Check initial class balance
fraud_count = df['Class'].value_counts()
print(f"\n📊 Initial class distribution:\nGenuine: {fraud_count[0]} | Fraud: {fraud_count[1]}")

# Step 3: Preprocess
df.drop('Time', axis=1, inplace=True)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

# Step 4: Visual - Class Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=y, palette='pastel')
plt.title("Transaction Class Count")
plt.xlabel("Class (0=Genuine, 1=Fraud)")
plt.tight_layout()
plt.show()

# Step 5: Apply SMOTE
print("\n⚖️ Applying SMOTE for class balancing...")
sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_resample(X, y)
print(f"✅ Balanced class distribution: Genuine={sum(y_res==0)}, Fraud={sum(y_res==1)}")

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=1)

# Step 7: Train Random Forest (quick)
print("\n🚀 Detecting fraud using Random Forest...")
model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# Step 8: Predict and Evaluate
y_pred = model.predict(X_test)

print("\n🔎 Evaluation Metrics:")
print(classification_report(y_test, y_pred))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

# Step 9: Visual - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 10: Visual - Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index, palette='rocket')
plt.title("Top 10 Features Important for Fraud Detection")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Step 11: Visual - Fraud vs Amount
plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=df, palette='Set3')
plt.title("Amount Distribution: Genuine vs Fraud")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.show()
