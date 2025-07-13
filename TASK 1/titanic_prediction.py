# Titanic Survival Prediction (Single CSV Version)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==== STEP 1: Load Data ====
train_path = "C:\\Users\\nanthinisenthil18\\Desktop\\titanic_prediction\\train.csv"
print("🚀 Loading dataset...")
df = pd.read_csv(train_path)
print("✅ Data loaded:", df.shape)

# ==== STEP 2: Preprocessing ====
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)

# Drop columns that don't help much
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Label encode categorical data
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ==== STEP 3: Train-Test Split ====
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== STEP 4: Train Model ====
print("🧠 Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ==== STEP 5: Evaluate Model ====
y_pred = model.predict(X_test)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ==== STEP 6: Visualize ====
# Actual survival count
plt.figure(figsize=(5,4))
y_test.value_counts().plot(kind='bar', color=['coral', 'skyblue'])
plt.title("Actual Survival Count")
plt.xticks([0, 1], ['Did Not Survive', 'Survived'], rotation=0)
plt.xlabel("Survival")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,4))
plt.barh(features, importances, color='seagreen')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

input("\n✅ Press Enter to exit...")
