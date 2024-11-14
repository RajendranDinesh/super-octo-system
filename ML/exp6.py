# Import necessary libraries
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate synthetic data to simulate social media users with different attributes
# 1000 samples, with 5 features each representing user characteristics
X, _ = make_blobs(
    n_samples=1000,    # Number of samples
    n_features=5,      # Features representing user characteristics
    centers=4,         # Initial number of centers (we'll validate this)
    cluster_std=1.0,   # Standard deviation of clusters
    random_state=42
)

# Finding the optimal number of clusters using the Elbow method
inertia = []   # List to hold the inertia values for each k
silhouette_scores = []  # List to hold silhouette scores for each k
K = range(2, 10)   # Range of clusters to try

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the Elbow curve for Inertia
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot the Silhouette Score to analyze clustering quality
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Optimal k')
plt.show()

# Choose the optimal number of clusters (e.g., based on the Elbow point or highest silhouette score)
optimal_k = 4  # For example, if we see k=4 has a good balance in the graphs
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(X)

# Final evaluation metrics
print(f"Optimal number of clusters (k): {optimal_k}")
print(f"Final Inertia: {kmeans_optimal.inertia_}")
print(f"Silhouette Score for k={optimal_k}: {silhouette_score(X, kmeans_optimal.labels_):.2f}")
