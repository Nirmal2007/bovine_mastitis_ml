import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_mastitis_forecasting_model():
    print("1. Loading hardware sensor dataset with Cow Info...")
    df = pd.read_csv("mastitis_dataset_with_cow_info.csv")
    df = df.sort_values(["Cow_ID", "Day"]).reset_index(drop=True)

    # 2. Impute sensor missing readings per cow
    sensor_features = [
        "Body_Temperature_C",
        "Env_Temperature_C",
        "Env_Humidity_Percent",
        "Cows_Movement_Activity",
        "Milk_Conductivity_mS_cm",
    ]
    df[sensor_features] = df.groupby("Cow_ID")[sensor_features].ffill().bfill()

    # 3. Time-series feature engineering (Rolling averages & standard deviations)
    print("2. Extracting time-series features (3-day & 7-day rolling windows)...")
    for col in sensor_features:
        df[f"{col}_roll3_mean"] = df.groupby("Cow_ID")[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df[f"{col}_roll7_mean"] = df.groupby("Cow_ID")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[f"{col}_roll7_std"] = (
            df.groupby("Cow_ID")[col]
            .transform(lambda x: x.rolling(7, min_periods=1).std())
            .fillna(0)
        )

    # 4. Handle Categorical Cow Info (One-Hot Encoding Breed)
    print("3. Encoding categorical features (Breed)...")
    # This creates columns like 'Breed_Gir', 'Breed_Jersey', etc. with 0s and 1s
    df = pd.get_dummies(df, columns=["Breed"], drop_first=False, dtype=int)

    # 5. Subject-wise train/test split (80% Train Cows / 20% Unseen Test Cows)
    unique_cows = df["Cow_ID"].unique()
    train_cows = unique_cows[:800]
    test_cows = unique_cows[800:]

    train_df = df[df["Cow_ID"].isin(train_cows)]
    test_df = df[df["Cow_ID"].isin(test_cows)]

    ignore_cols = [
        "Cow_ID",
        "Day",
        "Mastitis_Status",
        "Mastitis_Next_7_Days",
        "Mastitis_Next_14_Days",
    ]

    # Feature columns automatically pick up Age_Years, all encoded Breed columns, and sensor data
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X_train, y_train = train_df[feature_cols], train_df["Mastitis_Next_7_Days"]
    X_test, y_test = test_df[feature_cols], test_df["Mastitis_Next_7_Days"]

    # 6. Train Model
    print("4. Training Gradient Boosting Model...")
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 7. Evaluation
    print("5. Evaluating model on 200 unseen cows...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.50).astype(int)

    print(f"\n--- Performance Metrics ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}\n")
    print(
        classification_report(
            y_test, preds, target_names=["Healthy", "Mastitis_Risk_7_Days"]
        )
    )

    # 8. Save pipeline artifacts
    joblib.dump(
        {"model": model, "feature_names": feature_cols}, "mastitis_model.pkl"
    )
    print("Model artifact successfully saved to 'mastitis_model.pkl'")


if __name__ == "__main__":
    train_mastitis_forecasting_model()