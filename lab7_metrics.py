import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split

def compute_metrics(input_path="../data/clean_dataset.csv", artifacts_path="../artifacts/"):
    df = pd.read_csv(input_path)
    
    print("--- Lab 7: Performance Metrics ---")
    features = ["lpi", "gdp_growth", "inflation", "conflict_intensity", "military_expenditure"]
    X = df[features]
    y_reg = df["business_viability"]
    
    # Binarize for Accuracy and F1-Score evaluation
    y_cls = (y_reg > 50.0).astype(int) 
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    
    # 1. Linear Regression for "Business Success Ratio"
    lr = LinearRegression()
    lr.fit(X_train, y_train_reg)
    success_ratio_preds = lr.predict(X_test)
    joblib.dump(lr, os.path.join(artifacts_path, "linear_regression_model.joblib"))
    print(f"Linear Regression (Business Success Ratio) R2: {r2_score(y_test_reg, success_ratio_preds):.4f}\n")
    
    # Evaluate previously trained baselines
    models_to_eval = ["knn_model.joblib", "svm_model.joblib", "randomforest_model.joblib", "decisiontree_model.joblib"]
    
    for model_file in models_to_eval:
        model_path = os.path.join(artifacts_path, model_file)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            
            r2 = r2_score(y_test_reg, preds)
            
            cls_preds = (preds > 50.0).astype(int)
            acc = accuracy_score(y_test_cls, cls_preds)
            f1 = f1_score(y_test_cls, cls_preds)
            
            name = model_file.split('_')[0].upper()
            print(f"[{name}]")
            print(f"  R2-Score : {r2:.4f}")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1-Score : {f1:.4f}\n")
        else:
            print(f"Warning: {model_path} not found.")

if __name__ == "__main__":
    compute_metrics()
