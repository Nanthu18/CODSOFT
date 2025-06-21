# 📌 Task 4: Sales Prediction using Advertising.csv 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Step 1: Load dataset
df = pd.read_csv('Advertising.csv')
print(df.head())

# 🔹 Step 2: Show average sales per category (TV, Radio, Newspaper)
avg_sales = {
    'TV': df.groupby(pd.cut(df['TV'], bins=5), observed=True)['Sales'].mean(),
    'Radio': df.groupby(pd.cut(df['Radio'], bins=5), observed=True)['Sales'].mean(),
    'Newspaper': df.groupby(pd.cut(df['Newspaper'], bins=5), observed=True)['Sales'].mean()
}

# 🔹 Step 3: Plot average sales for each category one at a time

# TV vs Sales
plt.figure(figsize=(6, 4))
avg_sales['TV'].plot(kind='bar', color='coral')
plt.title("Average Sales by TV Ad Spend Range")
plt.ylabel("Sales")
plt.xlabel("TV Spend Range")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Radio vs Sales
plt.figure(figsize=(6, 4))
avg_sales['Radio'].plot(kind='bar', color='seagreen')
plt.title("Average Sales by Radio Ad Spend Range")
plt.ylabel("Sales")
plt.xlabel("Radio Spend Range")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Newspaper vs Sales
plt.figure(figsize=(6, 4))
avg_sales['Newspaper'].plot(kind='bar', color='mediumpurple')
plt.title("Average Sales by Newspaper Ad Spend Range")
plt.ylabel("Sales")
plt.xlabel("Newspaper Spend Range")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔹 Step 4: Prepare data for model
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 Step 6: Predictions
y_pred = model.predict(X_test)

# 🔹 Step 7: Evaluation
print("\nR² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 🔹 Step 8: Plot Actual vs Predicted Sales
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔹 Step 9: Combined Sales Comparison by Channel (Bar + Pie)

# Calculate average sales across all ranges for each channel
channel_means = {
    'TV': avg_sales['TV'].mean(),
    'Radio': avg_sales['Radio'].mean(),
    'Newspaper': avg_sales['Newspaper'].mean()
}

# 📊 Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(channel_means.keys(), channel_means.values(), color=['coral', 'seagreen', 'mediumpurple'])
plt.title("Average Sales Comparison by Advertising Channel")
plt.ylabel("Average Sales")
plt.xlabel("Channel")
plt.tight_layout()
plt.show()

# 🥧 Pie Chart 
plt.figure(figsize=(6, 6))
plt.pie(channel_means.values(), labels=channel_means.keys(), autopct='%1.1f%%',
        colors=['coral', 'seagreen', 'mediumpurple'], startangle=140)
plt.title("Sales Distribution by Advertising Channel")
plt.tight_layout()
plt.show()
