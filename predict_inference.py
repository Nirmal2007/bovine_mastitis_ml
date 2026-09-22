import pandas as pd
import joblib
import numpy as np

# 1. Load the pre-trained model and encoder
try:
    model = joblib.load('mastitis_risk_model.pkl')
    encoder = joblib.load('mastitis_encoder.pkl')
except FileNotFoundError:
    print("Error: Train the model first to generate the .pkl files.")
    exit()

# 2. Incoming hardware data dictionary for a HEALTHY cow (Last 7 Days)
new_cow_data = {
    'Age': 4,
    'Breed': 'Holstein',
    'Farm_Cleanliness': 'Yes',  # Clean barn hygiene

    # Day 1 (7 Days Ago) - Baseline metrics
    'env_temp_day1': 22.0, 'env_humidity_day1': 55.0, 'body_temp_day1': 38.3,
    'cows_movement_day1': 4800, 'milk_conductivity_day1': 4.1, 'milk_yield_day1': 25.0,

    # Day 2
    'env_temp_day2': 22.5, 'env_humidity_day2': 56.0, 'body_temp_day2': 38.4,
    'cows_movement_day2': 4900, 'milk_conductivity_day2': 4.2, 'milk_yield_day2': 25.5,

    # Day 3
    'env_temp_day3': 21.8, 'env_humidity_day3': 54.0, 'body_temp_day3': 38.3,
    'cows_movement_day3': 4750, 'milk_conductivity_day3': 4.1, 'milk_yield_day3': 24.8,

    # Day 4
    'env_temp_day4': 23.0, 'env_humidity_day4': 58.0, 'body_temp_day4': 38.5,
    'cows_movement_day4': 5000, 'milk_conductivity_day4': 4.2, 'milk_yield_day4': 25.2,

    # Day 5
    'env_temp_day5': 22.2, 'env_humidity_day5': 57.0, 'body_temp_day5': 38.4,
    'cows_movement_day5': 4850, 'milk_conductivity_day5': 4.3, 'milk_yield_day5': 25.0,

    # Day 6
    'env_temp_day6': 21.5, 'env_humidity_day6': 55.0, 'body_temp_day6': 38.3,
    'cows_movement_day6': 4950, 'milk_conductivity_day6': 4.2, 'milk_yield_day6': 25.4,

    # Day 7 (Today)
    'env_temp_day7': 22.0, 'env_humidity_day7': 56.0, 'body_temp_day7': 38.4,
    'cows_movement_day7': 5100, 'milk_conductivity_day7': 4.2, 'milk_yield_day7': 25.1
}

# 3. Convert dictionary to a DataFrame (1 row)
df_new = pd.DataFrame([new_cow_data])

# 4. Apply the exact same encoding used during training
categorical_cols = ['Breed', 'Farm_Cleanliness']
df_new[categorical_cols] = encoder.transform(df_new[categorical_cols])

# 5. Run Prediction
risk_prediction = model.predict(df_new)[0]

# Clip value between 0 and 100% just in case of statistical outliers
risk_prediction = np.clip(risk_prediction, 0, 100)

print(f"\n🔔 Mastitis Infection Risk (Next 7-14 Days): {risk_prediction:.2f}%")

if risk_prediction > 75.0:
    print("STATUS: CRITICAL. Isolate cow immediately and consult veterinary diagnostic tools.")
elif risk_prediction > 40.0:
    print("STATUS: WATCH. Monitor temperature and milk conductivity closely.")
else:
    print("STATUS: HEALTHY. Normal behavioral patterns detected.")