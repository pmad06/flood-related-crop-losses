import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('practice_by_county_final(in).csv')

# checks to verify csv is loaded properly
print(df.isna().sum())
print(df['Total_Indemnity'].describe())

# column being used for clustering
X = df[['Total_Indemnity']].values

# scaling the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# run k-means clustering for k values from 2 to 8 and store the results
results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    results.append({
        'k': k,
        # elbow method = determines how tightly the clusters are packed together
        # lower inertia = tightly packing = more clusters since each point can easily 
        # belong to a cluster
        'inertia': kmeans.inertia_,
        # silhouette method = measures how similar a point is to its own cluster compared to other clusters
        # high silhouette score = well separted clusters
        'silhouette_score': silhouette_score(X_scaled, labels)
    })

results_df = pd.DataFrame(results)
print(results_df)

# determine the best k based on silhouette score
best_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
print(f'Best number of clusters based on silhouette score: {best_k}')

# check to ensure there is a useful split based on the top silhouette score
print(df[['County_Name', 'Total_Indemnity']].sort_values('Total_Indemnity', ascending=False).head(10))

# manually chosen k to get a useful split (Low/Moderate/High)
chosen_k = 3
final_kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
df['cluster'] = final_kmeans.fit_predict(X_scaled)

# summary statistics for each cluster
print(df.groupby('cluster')['Total_Indemnity'].describe())

# print counties in each cluster sorted by Total_Indemnity
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c].sort_values('Total_Indemnity', ascending=False)
    print(f"\n--- Cluster {c} (n={len(sub)}) ---")
    print(sub[['County_Name', 'Total_Indemnity']].to_string(index=False))

# plot elbow and silhouette results, side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(results_df['k'], results_df['inertia'], marker='o')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')

axes[1].plot(results_df['k'], results_df['silhouette_score'], marker='o', color='orange')
axes[1].set_title('Silhouette Method')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette Score')

# save and display chart
plt.tight_layout()
plt.savefig('kmeans_analysis_results.png')
plt.show()

# export csv file with clusters 
df.to_csv('practice_by_county_final_with_clusters.csv', index=False)    