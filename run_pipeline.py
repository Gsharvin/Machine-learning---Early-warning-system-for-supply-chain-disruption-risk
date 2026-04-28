import dataset_generator
import lab1_preprocessing
import lab2_to_5_predictive
import lab6_unsupervised
import lab7_metrics
import os

if __name__ == "__main__":
    print("========================================")
    print("🚀 NEPTUNE ML LAB PIPELINE STARTING 🚀")
    print("========================================")
    
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../artifacts", exist_ok=True)
    
    # Lab 0: Data Gen
    print("\n[Lab 0] Generating Dataset...")
    dataset_generator.generate_dataset()
    
    # Lab 1: Preprocessing
    print("\n[Lab 1] Preprocessing data...")
    lab1_preprocessing.preprocess_data()
    
    # Labs 2-5: Predictive Engine
    print("\n[Labs 2-5] Training Predictive Models...")
    lab2_to_5_predictive.train_predictive_models()
    
    # Lab 6: Unsupervised Learning
    print("\n[Lab 6] Training DBSCAN...")
    lab6_unsupervised.train_unsupervised_models()
    
    # Lab 7: Metrics
    print("\n[Lab 7] Evaluating Model Metrics...")
    lab7_metrics.compute_metrics()
    
    print("\n========================================")
    print("✅ NEPTUNE ML LAB PIPELINE COMPLETE ✅")
    print("========================================")
