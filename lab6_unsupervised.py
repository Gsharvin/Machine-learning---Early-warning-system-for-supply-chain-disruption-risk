import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import DBSCAN

def train_unsupervised_models(input_path="../data/clean_dataset.csv", output_path="../data/clustered_dataset.csv", artifacts_path="../artifacts/"):
    df = pd.read_csv(input_path)
    os.makedirs(artifacts_path, exist_ok=True)
    
    print("--- Lab 6: Unsupervised Learning (DBSCAN) ---")
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    X = df[features]
    
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    df["risk_cluster"] = clusters
    df.to_csv(output_path, index=False)
    
    # Note: DBSCAN does not predict on new samples in scikit-learn standardly 
    # but we can save the model for reference or if we want to extract core samples
    joblib.dump(dbscan, os.path.join(artifacts_path, "dbscan_model.joblib"))
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    noise_points = list(clusters).count(-1)
    
    print(f"DBSCAN identified {n_clusters} Geopolitical Risk Hubs (clusters) and {noise_points} noise points.")
    print(f"Clustered dataset saved with 'risk_cluster' to {output_path}")

if __name__ == "__main__":
    train_unsupervised_models()
