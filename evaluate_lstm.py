import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def create_sequences(df, history_size=5, future_size=10):
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    targets = ["business_viability", "supply_chain_stability"]
    X, y = [], []
    for country, group in df.groupby("country"):
        group = group.sort_values("year")
        if len(group) < history_size + future_size: continue
        feature_vals = group[features].values
        target_vals = group[targets].values
        for i in range(len(group) - history_size - future_size + 1):
            X.append(feature_vals[i : i + history_size])
            y.append(target_vals[i + history_size : i + history_size + future_size].flatten())
    return np.array(X), np.array(y)

# Load data
df = pd.read_csv("data/clean_dataset.csv")
X, y = create_sequences(df)

# Split (same seed as training if possible, but 42 is standard)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model_path = "artifacts/lstm_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    preds = model.predict(X_test)
    
    # We evaluate Year 1 Business Viability (Index 0)
    # Threshold at 0.0 because data is StandardScaled? 
    # Wait, lab7_metrics used (y_reg > 50.0). 
    # But clean_dataset features are scaled. Targets are NOT scaled in lab1_preprocessing.py.
    # Let's check Lab 1 again. 
    # It says: "Normalize features, but also scale the targets so models work nicely, but we'll save separate scalers."
    # Wait, line 36: df_features_scaled = scaler.fit_transform(df[features]); df[features] = df_features_scaled
    # It doesn't scale "business_viability".
    
    # Binarize
    y_test_cls = (y_test[:, 0] > 50.0).astype(int)
    preds_cls = (preds[:, 0] > 50.0).astype(int)
    
    acc = accuracy_score(y_test_cls, preds_cls)
    prec = precision_score(y_test_cls, preds_cls)
    rec = recall_score(y_test_cls, preds_cls)
    f1 = f1_score(y_test_cls, preds_cls)
    
    print(f"LSTM Year-1 Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
else:
    print("Model not found")
