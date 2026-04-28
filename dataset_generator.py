import pandas as pd
import numpy as np
import os

def generate_dataset(output_path="../data/raw_dataset.csv"):
    np.random.seed(42)
    # Using 215 countries
    countries = [f"Country_{i}" for i in range(1, 216)]
    years = list(range(2010, 2027))
    
    data = []
    
    for country in countries:
        # Base stats for the country
        base_lpi = np.random.uniform(1.5, 4.5)
        base_gdp = np.random.normal(3.0, 2.0)
        base_inflation = np.abs(np.random.normal(3.0, 3.0))
        base_conflict = np.random.uniform(0, 8)
        base_military = np.random.uniform(1.0, 5.0)
        
        for year in years:
            # Random walks and noise
            gdp_growth = base_gdp + np.random.normal(0, 1.5)
            inflation = base_inflation + np.random.normal(0, 0.5)
            lpi = max(1.0, min(5.0, base_lpi + np.random.normal(0, 0.2)))
            conflict_intensity = max(0, min(10, base_conflict + np.random.normal(0, 1.0)))
            military_expenditure = max(0.1, base_military + np.random.normal(0, 0.2))
            
            # Simulated target properties for future ML models
            # "Business Viability" (0-100) -> Higher LPI and GDP, lower inflation/conflict
            viability = (lpi / 5.0) * 40 + (max(0, gdp_growth) / 10.0) * 30 - (inflation / 20.0) * 15 - (conflict_intensity / 10.0) * 15
            viability = max(0, min(100, viability + 50 + np.random.normal(0, 5)))
            
            # "Supply Chain Stability" (0-100) -> Heavily dependent on conflict and LPI
            stability = (lpi / 5.0) * 60 - (conflict_intensity / 10.0) * 40
            stability = max(0, min(100, stability + 50 + np.random.normal(0, 5)))

            # Inject missing data (approx 5%)
            if np.random.rand() < 0.05: lpi = np.nan
            if np.random.rand() < 0.05: gdp_growth = np.nan
            if np.random.rand() < 0.05: inflation = np.nan
                
            # Inject outliers (approx 1%)
            if np.random.rand() < 0.01: inflation = inflation * 10
            if np.random.rand() < 0.01: conflict_intensity = conflict_intensity * 2

            data.append({
                "country": country,
                "year": year,
                "lpi": lpi,
                "gdp_growth": gdp_growth,
                "inflation": inflation,
                "conflict_intensity": conflict_intensity,
                "military_expenditure": military_expenditure,
                "business_viability": viability,
                "supply_chain_stability": stability
            })
            
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset: {output_path} | Shape: {df.shape}")

if __name__ == "__main__":
    generate_dataset()
