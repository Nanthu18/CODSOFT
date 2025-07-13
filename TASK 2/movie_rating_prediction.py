# movie_rating_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
print("✅ Dataset Loaded! Shape:", df.shape)

# STEP 2: View some data
print("\n🎬 Sample Data:")
print(df.head())

# STEP 3: Drop unused columns
df.drop(columns=['Poster_Link', 'Overview', 'Votes'], errors='ignore', inplace=True)

# STEP 4: Clean missing data
df.dropna(subset=['Rating'], inplace=True)
df.fillna(method='ffill', inplace=True)

# STEP 5: Label encode categorical columns
le = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# STEP 6: Feature Engineering - Convert 'Duration' to number
if 'Duration' in df.columns:
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)

# STEP 7: Define features & target
features = ['Genre', 'Director', 'Actor 1', 'Actor 2']
if 'Duration' in df.columns:
    features.append('Duration')
X = df[features]
y = df['Rating']

# STEP 8: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 9: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 10: Predictions
y_pred = model.predict(X_test)

# STEP 11: Evaluation
print("\n📊 Model Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MSE     : {mean_squared_error(y_test, y_pred):.2f}")

# STEP 12: Actual vs Predicted (Scatter)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal')
plt.plot([0, 10], [0, 10], color='red', linestyle='--')
plt.title("Actual vs Predicted Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 13: Feature importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=features, y=importances, palette='rocket')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# STEP 14: New Chart – Rating distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Rating'], bins=20, kde=True, color='coral')
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.show()

# STEP 15: New Chart – Genre vs Average Rating
plt.figure(figsize=(8, 4))
genre_map = dict(zip(df['Genre'], le.inverse_transform(df['Genre'])))
df['Genre_Label'] = df['Genre'].map(genre_map)
avg_genre_rating = df.groupby('Genre_Label')['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=avg_genre_rating.index, y=avg_genre_rating.values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Average IMDb Rating per Genre")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# STEP 16: Save predictions to CSV 
results_df = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
results_df.to_csv("predicted_movie_ratings.csv", index=False)
print("✅ Predictions saved to 'predicted_movie_ratings.csv'")
