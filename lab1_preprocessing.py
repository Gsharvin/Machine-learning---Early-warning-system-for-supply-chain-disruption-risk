import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(input_path="../data/raw_dataset.csv", output_path="../data/clean_dataset.csv", scaler_path="../artifacts/scaler.joblib"):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Lab 1: Median imputation for missing data
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    for col in features:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {col} missing values with median: {median_val:.4f}")
            
    # Lab 1: IQR-based outlier filtering
    # Instead of dropping rows (which breaks time-series sequences for LSTM), we clip them to bounds.
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    print("Outliers clipped using IQR bounds.")

    # Lab 1: StandardScaler for feature normalization
    scaler = StandardScaler()
    # Normalize features, but also scale the targets so models work nicely, 
    # but we'll save separate scalers.
    df_features_scaled = scaler.fit_transform(df[features])
    df[features] = df_features_scaled
    
    # Save the scaler using joblib
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    print(f"Cleaned dataset saved to {output_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    preprocess_data()
