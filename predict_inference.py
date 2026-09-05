import joblib
import pandas as pd
import numpy as np


def showcase_mastitis_prediction():
    print("=" * 60)
    print("  MASTITIS ML MODEL - PREDICTION SHOWCASE")
    print("=" * 60)

    # 1. Load the trained model artifacts
    model_file = "mastitis_model.pkl"
    dataset_csv = "mastitis_dataset_with_cow_info.csv"

    try:
        artifacts = joblib.load(model_file)
        if isinstance(artifacts, dict):
            model = artifacts["model"]
            feature_names = artifacts["feature_names"]
        else:
            model = artifacts
            feature_names = getattr(model, "feature_names_in_", None)
        print(f"[SUCCESS] Loaded model and features from '{model_file}'")
    except Exception as e:
        print(f"[ERROR] Could not load model file '{model_file}': {e}")
        return

    # 2. Load dataset metadata (to grab valid cow ages and breeds)
    try:
        df_dataset = pd.read_csv(dataset_csv)
        print(f"[SUCCESS] Loaded dataset reference from '{dataset_csv}'")
    except Exception as e:
        print(f"[ERROR] Could not load dataset CSV: {e}")
        return

    # 3. Create a sample live reading simulating sensor telemetry for a specific cow
    # (Notice how Cows_Movement_Activity is within the trained dataset range: ~3000-5000)
    sample_cow_id = "C0001"
    cow_meta = (
        df_dataset[df_dataset["Cow_ID"] == sample_cow_id]
        [["Age_Years", "Breed"]]
        .drop_duplicates()
        .iloc[0]
    )

    age = cow_meta["Age_Years"]
    breed = cow_meta["Breed"]

    print(f"\n--- Testing Subject Profile ---")
    print(f"Cow ID : {sample_cow_id}")
    print(f"Breed  : {breed}")
    print(f"Age    : {age} years")

    # Let's simulate a rolling window buffer of recent sensor readings (e.g., 7 days or recent intervals)
    # We create a small 7-row dataframe to mimic rolling features correctly
    simulated_sensor_data = {
        "Body_Temperature_C": [38.8, 38.9, 39.0, 39.1, 39.3, 39.5, 39.8],  # Rising temperature (risk indicator)
        "Env_Temperature_C": [28.0, 28.5, 29.0, 28.5, 28.0, 29.1, 28.8],
        "Env_Humidity_Percent": [65.0, 66.0, 64.0, 65.0, 68.0, 70.0, 67.0],
        "Cows_Movement_Activity": [3800, 3750, 3700, 3500, 3200, 2800, 2400],  # Dropping activity (lethargy)
        "Milk_Conductivity_mS_cm": [4.6, 4.7, 4.8, 4.0, 4.3, 4.8, 4.4],  # Rising milk EC/TDS (classic mastitis signal)
    }

    df_live = pd.DataFrame(simulated_sensor_data)

    print("\n--- Live Incoming Sensor Stream (Latest Window) ---")
    print(df_live.tail(3).to_string(index=False))

    # 4. Apply Time-Series Feature Engineering (Matching training script)
    sensor_cols = [
        "Body_Temperature_C",
        "Env_Temperature_C",
        "Env_Humidity_Percent",
        "Cows_Movement_Activity",
        "Milk_Conductivity_mS_cm",
    ]

    for col in sensor_cols:
        df_live[f"{col}_roll3_mean"] = df_live[col].rolling(3, min_periods=1).mean()
        df_live[f"{col}_roll7_mean"] = df_live[col].rolling(7, min_periods=1).mean()
        df_live[f"{col}_roll7_std"] = df_live[col].rolling(7, min_periods=1).std().fillna(0)

    # 5. Add Metadata & One-Hot Encoding for Breed
    df_live["Age_Years"] = age

    for col in feature_names:
        if col.startswith("Breed_"):
            df_live[col] = 1 if col == f"Breed_{breed}" else 0
        elif col not in df_live.columns:
            df_live[col] = 0

    # Extract the absolute latest row for final prediction
    X_inference = df_live.iloc[[-1]][feature_names]

    # 6. Predict Risk Probability
    print("\n--- Running ML Inference ---")
    probabilities = model.predict_proba(X_inference)
    risk_probability = probabilities[0][1]  # Probability of class 1 (Mastitis Risk in 7 Days)
    risk_percentage = round(risk_probability * 100, 2)

    print(f"Calculated 7-Day Mastitis Risk: {risk_percentage}%")

    # 7. Provide Status and Action Recommendation
    if risk_percentage > 60.0:
        status = "HIGH RISK - Inspect Immediately"
        action = "Isolate subject. Conduct California Mastitis Test (CMT) and consult vet."
    elif risk_percentage > 35.0:
        status = "MODERATE RISK - Monitor Closely"
        action = "Manually check udder for heat or swelling. Strip milk to check for clots."
    else:
        status = "LOW RISK - Normal"
        action = "Maintain standard hygiene and post-milking teat dipping."

    print(f"Status Recommendation  : {status}")
    print(f"Action Plan            : {action}")
    print("=" * 60)


if __name__ == "__main__":
    showcase_mastitis_prediction()