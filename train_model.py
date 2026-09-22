import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 1. Load Data
df = pd.read_csv("synthetic_bovine_mastitis_risk_1Lakh.csv")

# 2. Separate Features (X) and Target Risk Percentage (y)
X = df.drop(columns=['Cow_ID', 'risk_percentage'])
y = df['risk_percentage']

# 3. Encode Categorical Data (Breed, Farm_Cleanliness)
categorical_cols = ['Breed', 'Farm_Cleanliness']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize the Gradient Boosting Algorithm
# Tell the model which columns are categorical so it processes them optimally
cat_feature_indices = [X.columns.get_loc(c) for c in categorical_cols]

print("Training Gradient Boosting Model on 80,000 rows...")
model = HistGradientBoostingRegressor(
    max_iter=400,            # Number of boosting stages/trees
    learning_rate=0.05,      # Step size for learning
    max_leaf_nodes=63,       # Complexity of each tree
    categorical_features=cat_feature_indices,
    random_state=42
)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Evaluate Performance
y_pred = model.predict(X_test)
print(f"--- Model Performance ---")
print(f"Mean Absolute Error (MAE): \t {mean_absolute_error(y_test, y_pred):.2f}%")
print(f"Root Mean Squared Error (RMSE):  {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}%")
print(f"R2 Score (Accuracy): \t\t {r2_score(y_test, y_pred) * 100:.2f}%")

# 8. Save Model and Encoder for real-time predictions
joblib.dump(model, 'mastitis_risk_model.pkl')
joblib.dump(encoder, 'mastitis_encoder.pkl')
print("Model and Encoder saved successfully.")