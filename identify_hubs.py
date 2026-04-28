import pandas as pd
import os

base_dir = os.getcwd()
df = pd.read_csv(os.path.join(base_dir, "data", "clustered_dataset.csv"))

# Get latest year data for each country
latest_df = df.sort_values(['country', 'year']).groupby('country').tail(1)

# Sort by LPI and Military
hubs = latest_df.sort_values('lpi', ascending=False).head(20)

print("Potential Global Hubs (Top 20 by LPI):")
print(hubs[['country', 'lpi', 'military_expenditure']].to_string())

# Find neighbors for Country_1 (as a test case)
from sklearn.neighbors import NearestNeighbors
features = ['lpi', 'gdp_growth', 'inflation', 'conflict_intensity', 'military_expenditure']
X = latest_df[features].fillna(0).values

nn = NearestNeighbors(n_neighbors=6)
nn.fit(X)

def get_peers(country_code):
    row = latest_df[latest_df['country'] == country_code]
    if row.empty: return []
    idx = latest_df.index.get_loc(row.index[0])
    distances, indices = nn.kneighbors([X[idx]])
    return latest_df.iloc[indices[0][1:]]['country'].tolist()

print("\nPeers for Country_1 (USA-like proxy):", get_peers('Country_9'))
print("Peers for Country_42 (India-like proxy):", get_peers('Country_42'))
