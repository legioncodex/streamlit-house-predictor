import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

# 1. LOAD & PREPARE DATA
print("Loading California Housing data...")
# Load data (target is in $100k, so we multiply to get actual USD)
data = fetch_california_housing(as_frame=True)
df = data.frame
df['MedHouseVal'] = df['MedHouseVal'] * 100000

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. BUILD THE PIPELINE
# A pipeline bundles preprocessing (StandardScaler) with the model (Ridge).
# 

# [Image of Machine Learning Pipeline diagram]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge()) # Using Ridge as suggested (Regularized Linear Model)
])

# 3. TRAIN & TUNE
print("Training and tuning the Ridge model...")
param_grid = {'model__alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Alpha found: {grid_search.best_params_['model__alpha']}")

# 4. SAVE THE PIPELINE
joblib.dump(best_model, 'house_price_model.joblib')
print("\nSuccess! Model saved to 'house_price_model.joblib'")