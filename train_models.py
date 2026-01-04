import pandas as pd
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- K-MEANS -----
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save K-Means model
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# ----- DBSCAN -----
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

# Save DBSCAN model
with open("dbscan_model.pkl", "wb") as f:
    pickle.dump(dbscan, f)

print("Models trained and saved successfully!")
