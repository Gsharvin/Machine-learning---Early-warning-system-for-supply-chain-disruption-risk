import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
# Ignore tensorflow warnings for a clean lab output
import logging
tf.get_logger().setLevel(logging.ERROR)

def create_sequences(df, history_size=5, future_size=10):
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    targets = ["business_viability", "supply_chain_stability"]
    
    X, y = [], []
    
    for country, group in df.groupby("country"):
        group = group.sort_values("year")
        feature_vals = group[features].values
        target_vals = group[targets].values
        
        for i in range(len(group) - history_size - future_size + 1):
            X.append(feature_vals[i : i + history_size])
            y.append(target_vals[i + history_size : i + history_size + future_size].flatten())
            
    return np.array(X), np.array(y)

def train_predictive_models(input_path="../data/clean_dataset.csv", artifacts_path="../artifacts/"):
    df = pd.read_csv(input_path)
    os.makedirs(artifacts_path, exist_ok=True)
    
    # 1. Baseline Models (Labs 2, 3, 4, 5)
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    X_base = df[features]
    y_base = df["business_viability"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
    
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(kernel="linear"),
        "RandomForest": RandomForestRegressor(n_estimators=20, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }
    
    print("--- Lab 2 to 5: Predictive Engine (Baselines) ---")
    for name, model in models.items():
        if name == "DecisionTree":
            # Lab: Implement Decision Trees with 5-Fold Cross-Validation.
            scores = cross_val_score(model, X_base, y_base, cv=5, scoring='r2')
            print(f"DecisionTree 5-Fold CV R2: {scores.mean():.4f}")
            
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} R2 Score on standard split: {score:.4f}")
        joblib.dump(model, os.path.join(artifacts_path, f"{name.lower()}_model.joblib"))

    # 2. LSTM Forecasting (Main Part)
    print("\n--- Lab: LSTM Forecasting ---")
    X_lstm, y_lstm = create_sequences(df, history_size=5, future_size=10)
    
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(5, 5), return_sequences=True),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(20) # (10 years) * (2 targets: viability, stability)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    if len(X_lstm) > 0:
        model.fit(X_lstm, y_lstm, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
        # tf model save compatible with TF 2.0+
        model.save(os.path.join(artifacts_path, "lstm_model.keras"))
        print("LSTM model trained and saved as 'lstm_model.keras'")
    else:
        print("Not enough sequences for LSTM training!")

if __name__ == "__main__":
    train_predictive_models()
