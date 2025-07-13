# 🌸 Iris Flower Classification
# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Step 1: Load the Dataset
df = pd.read_csv("IRIS.csv")  # Make sure IRIS.csv is in the same directory
print("First 5 rows of the dataset:")
print(df.head())

# 🔹 Step 2: Explore the Dataset
print("\nDataset Info:")
print(df.info())
print("\nClass distribution:")
print(df['species'].value_counts())

# 🔹 Step 3: Check for Missing Values
print("\nMissing values:\n", df.isnull().sum())

# 🔹 Step 4: Visualize the Data (Optional but helpful)
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 🔹 Step 5: Prepare Data for Model
X = df.drop('species', axis=1)
y = df['species']

# 🔹 Step 6: Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🔹 Step 7: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🔹 Step 8: Train a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 9: Predictions
y_pred = model.predict(X_test)

# 🔹 Step 10: Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 🔹 Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
